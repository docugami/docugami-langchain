# Adapted with thanks from https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb

import re
from typing import AsyncIterator, Dict, List, Optional, TypedDict

from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from docugami_langchain.base_runnable import BaseRunnable, TracedResponse
from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT
from docugami_langchain.params import RunnableParameters

PLAN_TASK_PROMPT = """For the following task, make plans that can solve the problem step by step. For each plan, indicate
which external tool together with tool input to retrieve evidence. You can store the evidence into a
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Supported tools can be ONLY be one of the following:
(1) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.
{tools}

Make sure you don't use a tool that does not exactly match one of the tools above.

Here is a general outline of a plan (using hypothetical tools, make sure the tools you actually use are the supported ones above)
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin! 
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}"""

SOLVE_TASK_PROMPT = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}
Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""

PLAN_REGEX_PATTERN = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"


class ReWOOState(TypedDict):
    steps: List
    plan_string: str
    task: str
    results: Dict
    result: str


class ReWOOAgent(BaseRunnable[ReWOOState]):
    """Agent that implements ReWOO (Reasoning WithOut Observation) https://arxiv.org/abs/2305.18323"""

    tools: list[BaseTool] = []

    def params(self) -> RunnableParameters:
        """The params are directly implemented in the runnable."""
        raise NotImplementedError()

    def prompt(
        self,
        params: RunnableParameters,
        num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
    ) -> BasePromptTemplate:
        """The prompt is directly implemented in the runnable."""
        raise NotImplementedError()

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """
        plan_prompt_template = ChatPromptTemplate.from_messages(
            [("human", PLAN_TASK_PROMPT)]
        )
        planner = plan_prompt_template | self.llm

        def get_plan(state: ReWOOState) -> Dict:
            task = state["task"]
            tool_list = ""
            for i in range(len(self.tools)):
                tool = self.tools[i]
                # Start from (2) since (1) is always the default LLM tool
                tool_list += f"({i + 2}) {tool.name}[input]: {tool.description}"
            result = planner.invoke({"task": task, "tools": tool_list})
            # Find all matches in the sample text
            matches = re.findall(PLAN_REGEX_PATTERN, result.content)
            return {"steps": matches, "plan_string": result.content}

        def get_current_task(state: ReWOOState) -> int:
            if state["results"] is None:
                return 1
            if len(state["results"]) == len(state["steps"]):
                return None  # type: ignore
            else:
                return len(state["results"]) + 1

        def tool_execution(state: ReWOOState) -> Dict:
            """Worker node that executes the tools of a given plan."""
            _step = get_current_task(state)
            _, step_name, tool, tool_input = state["steps"][_step - 1]
            _results = state["results"] or {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
            matching_tools = [
                t for t in self.tools if t.name.lower() == str(tool).lower()
            ]
            if matching_tools:
                result = matching_tools[0].invoke(tool_input)
            else:
                # if a tool name was made up, fall back to just llm
                result = self.llm.invoke(tool_input)

            _results[step_name] = str(result)
            return {"results": _results}

        def solve(state: ReWOOState) -> Dict:
            plan = ""
            for _plan, step_name, tool, tool_input in state["steps"]:
                _results = state["results"] or {}
                plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]\n\n"
                if step_name in _results:
                    plan += _results[step_name] + "\n"
            prompt = SOLVE_TASK_PROMPT.format(plan=plan, task=state["task"])
            result = self.llm.invoke(prompt)
            return {"result": result.content}

        def route(state: ReWOOState) -> str:
            _step = get_current_task(state)
            if _step is None:
                # We have executed all tasks
                return "solve"
            else:
                # We are still executing tasks, loop back to the "tool" node
                return "tool"

        graph = StateGraph(ReWOOState)
        graph.add_node("plan", get_plan)
        graph.add_node("tool", tool_execution)
        graph.add_node("solve", solve)
        graph.add_edge("plan", "tool")
        graph.add_edge("solve", END)
        graph.add_conditional_edges("tool", route)
        graph.set_entry_point("plan")

        return graph.compile()

    def run(  # type: ignore[override]
        self,
        task: str,
        config: Optional[dict] = None,
    ) -> ReWOOState:
        if not task:
            raise Exception("Input required: task")

        return super().run(
            task=task,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        task: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedResponse[ReWOOState]]:
        if not task:
            raise Exception("Input required: task")

        return super().run_stream(
            task=task,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[dict] = None,
    ) -> list[ReWOOState]:
        return super().run_batch(
            inputs=[{"task": i} for i in inputs],
            config=config,
        )
