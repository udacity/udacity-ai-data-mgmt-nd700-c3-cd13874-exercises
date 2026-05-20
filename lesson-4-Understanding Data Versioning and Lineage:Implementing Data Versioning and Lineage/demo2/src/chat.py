from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from common import load_params, load_secrets, get_embeddings, get_llm

PROMPT_PATH = Path("artifacts/system_prompt.txt")
CHROMA_DIR = Path("artifacts/chroma_db")


def main() -> None:
    params = load_params()
    secrets = load_secrets()

    embeddings = get_embeddings(params, secrets)
    llm = get_llm(params, secrets)

    vectorstore = Chroma(
        collection_name=params["rag"]["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": params["rag"]["k"]})
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            (
                "human",
                "Question: {question}\n\nRetrieved context:\n{context}\n\n"
                "Answer using only the retrieved context.",
            ),
        ]
    )

    print("Supermarket Supply Chain RAG Chat")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break

        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        messages = rag_prompt.format_messages(
            system_prompt=system_prompt,
            question=question,
            context=context,
        )

        response = llm.invoke(messages)

        print("\nAnswer:")
        print(response.content)
        print("\nRetrieved chunks:")
        for i, doc in enumerate(docs, start=1):
            print(f"{i}. metadata={doc.metadata}")
        print()


if __name__ == "__main__":
    main()