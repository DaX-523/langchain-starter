import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { CohereEmbeddings, ChatCohere } from "@langchain/cohere";
import { formatDocumentsAsString } from "langchain/util/document";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
// import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
);

const main = async () => {
  const docs = await loader.load();
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splits = await textSplitter.splitDocuments(docs);
  const store = await MemoryVectorStore.fromDocuments(
    splits,
    new CohereEmbeddings({
      apiKey: process.env.COHERE_API_KEY,
      model: "embed-english-v2.0",
    })
  );
  const retriever = store.asRetriever();

  // const prompt = await pull("rlm/rag-prompt");
  const llm = new ChatCohere({
    apiKey: process.env.COHERE_API_KEY,
    temperature: 0,
  });

  // const ragChain = await createStuffDocumentsChain({
  //   llm,
  //   prompt,
  //   outputParser: new StringOutputParser(),
  // });
  // console.log(
  //   prompt.promptMessages.map((msg) => msg.prompt.template).join("\n")
  // );
  // const res = await ragChain.invoke({
  //   context: await retriever.invoke("What is Task Decomposition?"),
  //   question: "What is Task Decomposition?",
  // });
  // console.log(res);
  const contextualizeQSystemPrompt = `Given a chat history and the latest user question
  which might reference context in the chat history, formulate a standalone question
  which can be understood without the chat history. Do NOT answer the question,
  just reformulate it.`;

  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);
  const contextualizeQChain = contextualizeQPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());

  const qaSystemPrompt = `You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    
    {context}`;
  const qaSystemTemplate = ChatPromptTemplate.fromMessages([
    ["system", qaSystemPrompt],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);

  const contextualizedQuestion = (input) => {
    if ("history" in input) {
      return contextualizeQChain;
    }
    return input.question;
  };

  const ragChain3 = RunnableSequence.from([
    RunnablePassthrough.assign({
      context: (input) => {
        if ("history" in input) {
          const chain = contextualizedQuestion(input);
          return chain.pipe(retriever).pipe(formatDocumentsAsString);
        }
        return "";
      },
    }),
    qaSystemTemplate,
    llm,
  ]);

  // const contextualQuestion = (input) => {
  //   if ("history" in input) {
  //     return contextualizeQChain;
  //   }
  //   return input.input;
  // };

  // const ragChain2 = RunnableSequence.from([
  //   RunnablePassthrough.assign({
  //     context: (input) => {
  //       if ("history" in input) {
  //         const chain = contextualQuestion(input);
  //         chain.pipe(retriever).pipe(formatDocumentsAsString);
  //       }
  //       return "";
  //     },
  //   }),
  //   qaSystemTemplate,
  //   llm,
  // ]);

  let history = [];
  const q1 = await ragChain3.invoke({
    history,
    input: "What is caching?",
  });

  console.log(1, q1);
  history.push(q1);
  const q2 = await ragChain3.invoke({
    history,
    input: "What are common ways of doing it? And doing what exactly?",
  });

  console.log(q2);
  history.push(q2);

  // const res2 = await contextualizeQChain.invoke({
  //   history: [
  //     new HumanMessage({ content: "What does LLM stands for?" }),
  //     new AIMessage({ content: "Large Language Model" }),
  //   ],
  //   input: "What does large means?",
  // });

  // console.log(res2);
};

main();
