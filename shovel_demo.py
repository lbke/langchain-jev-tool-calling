# from typesafe_sdk import Noul, TypeSafeClient
import uuid

from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter
import requests
import os


@tool
def shovel(depth_meters: int):
    """
    Use to dig a hole
    """
    return f"Dug a {depth_meters} meters deep hole"


model = ChatOpenRouter(model="google/gemini-3.1-flash-lite")


def pick_tool(user_prompt):
    # TODO: no typesafe key yet, fallback to open router
    # with TypeSafeClient(model="jev-latest") as ts_client:
    #     tool_calls = ts_client.system_one(
    #         state=user_prompt,
    #         questions={shovel.name: Noul(instructions=shovel.description)},
    #     )
    # https://openrouter.ai/docs/guides/community/jev-tutorial
    response = requests.post(
        "https://openrouter.ai/api/alpha/decisions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "typesafe/jev-1.13",
            "state": {
                "user_prompt": user_prompt + "\nYou are tasked to pick the next relevant tools for the task (multiple tools can be used in parallel if necessary)"
            },
            "questions": {
                shovel.name: {
                    "type": "noul",
                    "instructions": shovel.description,
                    "criteria": {
                        "true": "Use this tool for next step",
                        "false": "Do not use this tool yet",
                    },
                },
            },
        },
    )
    return response.json()


def get_system_one_cost(res):
    return res["usage"]["cost"]


def get_ai_message_cost(res):
    return res.response_metadata["cost"]


def print_cost(cost):
    runs_per_dollar = int(1/cost)
    print(
        f"For 1 dollar, you could run this tool calling decision {runs_per_dollar} times (cost was {cost}$)")


def main():
    print("## Jev tool calling")
    # 1. Pick tool
    user_prompt = "Dig a hole."
    tool_calls_res = pick_tool(user_prompt)
    print(tool_calls_res)
    c1 = get_system_one_cost(tool_calls_res)
    tool_calls = {t for t, v in tool_calls_res["answers"].items(
    ) if v["type"] == "noul" and v["noul"] > 0.7}
    print(tool_calls)
    has_shovel = "shovel" in tool_calls
    # 2. generate args
    # NOTE: jev could be used to decide the value of closed arguments
    if has_shovel:
        args_generator = model.with_structured_output(
            shovel.get_input_schema(),
            # get cost metadata
            include_raw=True
        )
        tool_args = args_generator.invoke(
            "\n".join(
                [user_prompt, f"Generate args for tool_call {shovel.name}"]),
        )
        c2 = get_ai_message_cost(tool_args["raw"])
        tool_args_json = tool_args["parsed"].model_dump()
        # 3. call the tool
        tool_res = shovel.invoke(tool_args["parsed"])
        # 4. Last LLM call - needs to fake an AI message calling the tool
        tool_call_id = str(uuid.uuid4())
        ai_msg = AIMessage(content="", tool_calls=[
                           {"name": shovel.name, "args": tool_args_json, "id": tool_call_id}])
        tool_msg = ToolMessage(content=tool_res, tool_call_id=tool_call_id)
        # TODO: caching may not be good when using multiple models like so?
        final_res = model.invoke(
            [("user", user_prompt), ai_msg, tool_msg]
        )
        c3 = get_ai_message_cost(final_res)
        print_cost(c1+c2+c3)
        final_res.pretty_print()
    else:
        # No tool call, wrap it up
        final_res = model.invoke(user_prompt)
        final_res.pretty_print()

    print("## Versus direct LLM call")
    # vs direct call
    direct_call_model = model.bind_tools([shovel])
    direct_res = direct_call_model.invoke([("user", user_prompt)])
    c4 = get_ai_message_cost(direct_res)
    direct_shovel_call = [
        tc for tc in direct_res.tool_calls if tc["name"] == shovel.name].pop()
    direct_tool_res = shovel.invoke(direct_shovel_call["args"])
    direct_tool_msg = ToolMessage(
        content=direct_tool_res, tool_call_id=direct_shovel_call["id"])
    direct_final_res = model.invoke(
        [("user", user_prompt), direct_res, direct_tool_msg])
    c5 = get_ai_message_cost(direct_final_res)
    print_cost(c4+c5)
    direct_final_res.pretty_print()


if __name__ == "__main__":
    main()
