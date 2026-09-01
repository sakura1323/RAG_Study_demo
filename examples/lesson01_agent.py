"""第 1 课：不启动 FastAPI，直接在终端测试 Agent。"""

from backend.agent_service import SimpleAgent


def main() -> None:
    agent = SimpleAgent()
    print("输入 exit 结束程序。试试：123.5 加 88 等于多少？")
    while True:
        question = input("\n你：").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        print("Agent：", end="", flush=True)
        for chunk in agent.stream(question):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()

