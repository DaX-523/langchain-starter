import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { CohereEmbeddings, ChatCohere } from "@langchain/cohere";
import { createRetrieverTool } from "langchain/tools/retriever";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const searchTool = new TavilySearchResults();

const main = async () => {
  // const query = await searchTool.invoke(
  //   "What is the humidity level in kuala lumpur?"
  // );
  // console.log(query);
  const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/user_guide"
  );

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
  // const retrieverResult = await retriever.invoke("how to upload a dataset");
  // console.log(retrieverResult[0]);

  const retrieverTool = createRetrieverTool(retriever, {
    name: "langsmith_search",
    description:
      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    verbose: true,
  });

  const tools = [searchTool, retrieverTool];

  const llm = new ChatCohere({
    apiKey: process.env.COHERE_API_KEY,
    temperature: 0,
  });

  // Get the prompt to use - you can modify this!
  // If you want to see the prompt in full, you can at:
  // https://smith.langchain.com/hub/hwchase17/openai-functions-agent
  const prompt = await pull("hwchase17/openai-functions-agent");
  console.log(prompt);

  const agent = await createOpenAIFunctionsAgent({
    llm,
    tools,
    prompt,
  });

  const executer = new AgentExecutor({ agent, tools, verbose: true });
  let chat_history = [];

  const result1 = await executer.invoke({
    input: "hi i am kaydee!",
    chat_history,
  });
  console.log(1, result1);
  const result2 = await executer.invoke({
    input: "who am i?",
    chat_history: [
      new HumanMessage("hi i am kaydee!"),
      new AIMessage("Hi Kaydee! How can I help you today?"),
    ],
  });

  console.log(2, result2);

  const msgHistory = new ChatMessageHistory();
  const AgentWithHistory = new RunnableWithMessageHistory({
    runnable: executer,
    getMessageHistory: (sessionId) => msgHistory,
    historyMessagesKey: "chat_history",
    inputMessagesKey: "input",
  });

  const result3 = await AgentWithHistory.invoke(
    {
      input: "Hi i am dax",
    },
    {
      configurable: {
        sessionId: "foo",
      },
    }
  );

  console.log(3, result3);
  const result4 = await AgentWithHistory.invoke(
    {
      input: "who am i?",
    },
    {
      configurable: {
        sessionId: "foo",
      },
    }
  );

  console.log(4, result4);
};

main();
