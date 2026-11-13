# from typesafe_sdk import Noul, TypeSafeClient
import uuid

from langchain.agents import create_agent
from langchain.messages import (
    AIMessage,
    MessageLikeRepresentation,
    ToolMessage,
    HumanMessage,
)
from langchain.tools import tool
from langchain_core.tools import Tool
from langchain_openrouter import ChatOpenRouter
import requests
import os


@tool
def shovel(depth_meters: int):
    """
    Use to dig a hole or fill it back.
    """
    return f"Dug a {depth_meters} meters deep hole"


@tool
def watering_can(use_big: bool):
    """
    Use to water something.
    Assume the watering_can to be always full.
    You can pick the big or the small watering can.
    Small is default.
    """
    if use_big:
        return f"Used the {'big' if use_big else 'small'} watering can"


@tool
def plant_tree():
    """
    Put tree in hole.
    """
    return "Plant planted"


model = ChatOpenRouter(model="google/gemini-3.1-flash-lite")
big_model = ChatOpenRouter(model="google/gemini-3.5-flash")


def pick_tool(messages: list[MessageLikeRepresentation], tools: list[Tool]):
    # TODO: no typesafe key yet, fallback to open router
    # with TypeSafeClient(model="jev-latest") as ts_client:
    #     tool_calls = ts_client.system_one(
    #         state=user_prompt,
    #         questions={shovel.name: Noul(instructions=shovel.description)},
    #     )
    # https://openrouter.ai/docs/guides/community/jev-tutorial
    # TODO: we could also pick values for bounded decisions by detecting boolean or enums arguments in tools
    response = requests.post(
        "https://openrouter.ai/api/alpha/decisions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "typesafe/jev-1.13",
            "state": {
                "messages": [
                    (
                        "system",
                        "You are tasked to pick the next relevant tools for the task (multiple tools can be used in parallel if necessary) based on the conversation.",
                    )
                ]
                + messages
            },
            "questions": {
                tool.name: {
                    "type": "noul",
                    "instructions": tool.description,
                    "criteria": {
                        "true": "Use this tool for next step",
                        "false": "Do not use this tool yet",
                    },
                }
                for tool in tools
            },
        },
    )
    return response.json()


def get_system_one_cost(res):
    return res["usage"]["cost"]


def get_ai_message_cost(res):
    return res.response_metadata["cost"]


def print_cost(cost):
    runs_per_dollar = int(1 / cost)
    print(
        f"For 1 dollar, you could run this tool calling decision {runs_per_dollar} times (cost was {cost}$)"
    )
    print(f"Running this generation 1000 (1K) times would cost {cost * 1000}$")
    print(
        f"Running this generation 1000000 (1M) times would cost {cost * 1000000}$")


def agent_loop():
    agent = create_agent(
        model=model,
        tools=[shovel, watering_can, plant_tree],
        system_prompt="You are the best tree planter in the world",
    )
    res = agent.invoke({"messages": [("user", "Plant a tree")]})
    # TODO: get final cost of agent?
    # c5 = get_ai_message_cost(direct_final_res)
    # print_cost(c4 + c5)
    # direct_final_res.pretty_print()
    print(res)
    for msg in res["messages"]:
        msg.pretty_print()


def tool_calling_jev():
    """
    Runs a single step of tool_calling with Jev
    Handles parallel tool_calls, but not sequential (agent loop) yet
    """
    costs = []
    # 1. Pick tool
    system_prompt = "You are the best tree planter in the world"
    user_prompt = "Plant a tree"
    tools = [shovel, watering_can, plant_tree]
    tools_map = {t.name: t for t in tools}
    tool_calls_res = pick_tool([("user", user_prompt)], tools)
    print(tool_calls_res)
    costs.append(get_system_one_cost(tool_calls_res))
    selected_tools = {
        t
        for t, v in tool_calls_res["answers"].items()
        if v["type"] == "noul" and v["noul"] > 0.7
    }
    print(selected_tools)
    # 2. generate args
    # NOTE: jev could be used to decide the value of closed arguments
    tool_messages = []
    ai_msg_tool_calls = []
    for tool_name in selected_tools:
        tool = tools_map[tool_name]
        args_generator = model.with_structured_output(
            tool.get_input_schema(),
            # get cost metadata
            include_raw=True,
        )
        tool_args = args_generator.invoke(
            "\n".join(
                [user_prompt, f"Generate args for tool_call {shovel.name}"]),
        )
        costs.append(get_ai_message_cost(tool_args["raw"]))
        tool_args_json = tool_args["parsed"].model_dump()
        # 3. call the tool
        tool_res = shovel.invoke(tool_args["parsed"])
        # 4. Last LLM call - needs to fake an AI message calling the tool
        tool_call_id = str(uuid.uuid4())
        ai_msg_tool_calls.append(
            {"name": tool.name, "args": tool_args_json, "id": tool_call_id}
        )
        tool_msg = ToolMessage(content=tool_res, tool_call_id=tool_call_id)
        tool_messages.append(tool_msg)

    # TODO: caching may not be good when using multiple models like so?
    ai_msg = AIMessage(content="", tool_calls=ai_msg_tool_calls)
    msgs = [HumanMessage(user_prompt), ai_msg, *tool_messages]
    final_res = model.invoke(msgs)
    costs.append(get_ai_message_cost(final_res))
    print_cost(sum(costs))
    for msg in msgs:
        msg.pretty_print()
    final_res.pretty_print()


if __name__ == "__main__":
    print("### Tool calling with Jev")
    tool_calling_jev()
    # print("### Agent loop")
    # agent_loop()
