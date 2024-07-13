import { ChatOpenAI, OpenAI } from "@langchain/openai";
import { ConversationChain, loadQAChain } from "langchain/chains";
import { PromptTemplate } from "@langchain/core/prompts";

import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatCohere } from "@langchain/cohere";

const contextualizeQSystemPrompt = `Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.`;

const condenseQuestionTemplate = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

  Chat History:
  {history}
  Follow Up Input: {question}
  Standalone question:`;
const CONDENSE_QUESTION_PROMPT = PromptTemplate.fromTemplate(
  condenseQuestionTemplate
);

const answerTemplate = `Answer the question based only on the following context:
  {context}
  
  Question: {question}
  `;
const ANSWER_PROMPT = PromptTemplate.fromTemplate(answerTemplate);

const makeChain = async (vectorstore) => {
  const llm = new ChatCohere({
    apiKey: process.env.COHERE_API_KEY,
    temperature: 0,
  });

  console.log("wttttffffffffffffffffffffffffffffffffffffffffffffff");
  console.log("{input}");

  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", condenseQuestionTemplate],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);

  console.log(contextualizeQPrompt);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever: vectorstore.asRetriever(),
    rephrasePrompt: contextualizeQPrompt,
  });

  // Answer question
  const qaSystemPrompt = `
  You are an assistant for question-answering tasks. Use
  the following pieces of retrieved context to answer the
  question. If you don't know the answer, just say that you
  don't know.
  \n\n
  {context}`;

  const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", qaSystemPrompt],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);

  const docChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt,
  });

  const ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: docChain,
  });

  return ragChain;
};

export default makeChain;
