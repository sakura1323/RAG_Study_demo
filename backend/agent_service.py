"""Agent 服务：负责模型、工具、消息、记忆和调用方式。"""

import os
from collections.abc import AsyncIterator, Iterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from .config import ENV_PATH


load_dotenv(ENV_PATH)


@tool
def add(a: float, b: float) -> float:
    """计算两个数字的和。"""
    return a + b


class SimpleAgent:
    """封装 Agent，供终端和 FastAPI 复用。"""

    def __init__(self) -> None:
        self.model = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.7,
        )
        self.agent = create_agent(
            model=self.model,
            tools=[add],
            system_prompt=SystemMessage(
                content="你是一个简洁的中文助手。遇到加法必须调用 add 工具。"
            ),
            checkpointer=InMemorySaver(),
        )

    def invoke(self, question: str, thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        old_count = len(self.agent.get_state(config).values.get("messages", []))
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]}, config=config
        )

        used_tool = False
        for message in result["messages"][old_count:]:
            if isinstance(message, AIMessage) and message.tool_calls:
                used_tool = True
                for call in message.tool_calls:
                    print(f"[Agent] 调用工具：{call['name']}，参数：{call['args']}")
            elif isinstance(message, ToolMessage):
                print(f"[Tool] 执行结果：{message.content}")
        if not used_tool:
            print("[Agent] 未调用工具，由模型直接回答")
        return str(result["messages"][-1].content)

    def stream(self, question: str, thread_id: str = "default") -> Iterator[str]:
        config = {"configurable": {"thread_id": thread_id}}
        for message, _metadata in self.agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(message, AIMessageChunk) and isinstance(message.content, str):
                yield message.content

    def answer_with_context(self, question: str, documents: list[Document]) -> str:
        """让 DeepSeek 只根据检索到的知识库内容回答。"""
        messages = self._build_rag_messages(question, documents)
        response = self.model.invoke(messages)
        return str(response.content)

    def _build_rag_messages(
        self, question: str, documents: list[Document]
    ) -> list[SystemMessage | HumanMessage]:
        """把检索结果和用户问题组装成 RAG 消息。"""
        context = "\n\n".join(
            f"[资料 {index}]\n{document.page_content}"
            for index, document in enumerate(documents, start=1)
        )
        return [
            SystemMessage(
                content=(
                    "你是个人知识库助手。只根据参考资料回答；"
                    "如果资料中没有答案，就说知识库中没有找到相关信息。"
                )
            ),
            HumanMessage(content=f"参考资料：\n{context}\n\n用户问题：{question}"),
        ]

    async def astream_with_context(
        self, question: str, documents: list[Document]
    ) -> AsyncIterator[str]:
        """让 DeepSeek 根据知识库资料异步流式回答。"""
        messages = self._build_rag_messages(question, documents)
        async for chunk in self.model.astream(messages):
            if isinstance(chunk.content, str):
                yield chunk.content

    async def astream(
        self, question: str, thread_id: str = "default"
    ) -> AsyncIterator[str]:
        config = {"configurable": {"thread_id": thread_id}}
        async for message, _metadata in self.agent.astream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(message, AIMessageChunk) and isinstance(message.content, str):
                yield message.content

