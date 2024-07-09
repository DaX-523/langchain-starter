const app = require("express")();
const { HumanMessage, AIMessage } = require("@langchain/core/messages");
const { ChatMistralAI } = require("@langchain/mistralai");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const {
  RunnableWithMessageHistory,
  RunnablePassthrough,
  RunnableSequence,
} = require("@langchain/core/runnables");
const { InMemoryChatMessageHistory } = require("@langchain/core/chat_history");

const model = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0,
  apiKey: process.env.MISTRAL_API_KEY,
});

// Manual approach

// model
//   .invoke([
//     new HumanMessage({ content: "Hi! I'm Dax" }),
//     new AIMessage({ content: "Hello Dax! How can I assist you today?" }),
//     new HumanMessage({ content: "What's my name?" }),
//   ])
//   .then((response) => {
//     console.log(response);
//   })
//   .catch((error) => {
//     console.error(error);
//   });

const messageHistories = {};

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a helpful assistant who remembers all details the user shares with you.",
  ],
  ["placeholder", "{chat_history}"],
  ["human", "{input}"],
]);

const chain = RunnableSequence.from([
  RunnablePassthrough.assign({
    chat_history: ({ chat_history }) => chat_history.slice(-2),
  }),
  prompt,
  model,
]);

const withMessageHistory = new RunnableWithMessageHistory({
  runnable: chain,
  getMessageHistory: (sessionid) => {
    if (!messageHistories[sessionid])
      messageHistories[sessionid] = new InMemoryChatMessageHistory();
    return messageHistories[sessionid];
  },
  inputMessagesKey: "input",
  historyMessagesKey: "chat_history",
  // config: {
  //   configurable: {
  //   sessionid: "user1"
  //   }
  // }
});

const config = {
  configurable: {
    sessionId: "user1",
  },
};

async function main() {
  try {
    const response = await withMessageHistory.invoke(
      {
        input: "Hi! I'm DaX",
      },
      config
    );

    console.log(response);

    const followupResponse = await withMessageHistory.invoke(
      {
        input: "What's my name?",
      },
      {
        configurable: {
          sessionId: "user2",
        },
      }
    );

    console.log(followupResponse);
    // for streaming the response

    // const stream = await withMessageHistory.stream(
    //   { input: "Hi I am Dax" },
    //   config
    // );

    // for await (const chunk of stream) {
    //   console.log(chunk.content);
    // }
  } catch (error) {
    console.error("Error:", error);
  }
}

main();

app.listen(3301, () => console.log("Server is running on port 3301"));
