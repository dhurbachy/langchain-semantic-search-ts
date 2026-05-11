// import { Document } from "@langchain/core/documents";
// import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
// import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
// import { OpenAIEmbeddings,ChatOpenAI } from "@langchain/openai";
// import { FaissStore } from "@langchain/community/vectorstores/faiss";
// import * as fs from "node:fs";

// const VECTOR_STORE_PATH = "./nike_index";
// let vectorStore;
// import { OllamaEmbeddings } from "@langchain/ollama";
// import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
// // const document=[
// //     new Document({
// //         pageContent:'DOgs are great companions, known for their loyalty and friendliness',
// //         metadata:{source:'mamals-pets-doc'}
// //     }),
// //     new Document({
// //         pageContent:"Cats are independent pets that often enjoy their own space",
// //         metadata:{source:'mamals-pets-doc'}
// //     }),
// // ];

// const loader=new PDFLoader("./nke-10k-2023.pdf");
// const docs=await loader.load();
// console.log(docs.length);
// console.log(docs[0].pageContent,docs[0].metadata);
// const textSplitters=new RecursiveCharacterTextSplitter({
//     chunkSize:1000,
//     chunkOverlap:200,
// });
// const allSplits=await textSplitters.splitDocuments(docs);
// console.log(allSplits.length);
// const embeddings=new OllamaEmbeddings({
//     // model:'text-embedding-3-large'
//     model:'mxbai-embed-large',
//     baseUrl:'http://localhost:11434'
// });
// const vector1=await embeddings.embedQuery(allSplits[0].pageContent);
// const vector2=await embeddings.embedQuery(allSplits[1].pageContent);
// console.log(`Generated vectors of length ${vector1.length}\n`);
// console.log(vector1.slice(0, 10));
// // const vectorStore=new MemoryVectorStore(embeddings);
// // await vectorStore.addDocuments(allSplits);
// // const query = "What was Nike's revenue in 2023?";
// // const results = await vectorStore.similaritySearch(query, 3);
// // results.forEach((res, i) => {
// //     console.log(`\n--- Result ${i + 1} (Page ${res.metadata.loc.pageNumber}) ---`);
// //     console.log(res.pageContent);
// // });
// if (fs.existsSync(VECTOR_STORE_PATH)) {
//   console.log("Loading existing index... (Instant)");
//   vectorStore = await FaissStore.load(VECTOR_STORE_PATH, embeddings);
// } else {
//   console.log("Creating new index... (One-time slow process)");
//   vectorStore = await FaissStore.fromDocuments(allSplits, embeddings);
//   await vectorStore.save(VECTOR_STORE_PATH);
// }

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai"; // Use this for NVIDIA NIM
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/ollama";
import * as fs from "node:fs";
import * as dotenv from "dotenv";

dotenv.config();
import { Embeddings } from "@langchain/core/embeddings";

class NVIDIAEmbeddings extends Embeddings {
    constructor(
        private apiKey: string,
        private model: string
    ) {
        super({});
    }

    async embedDocuments(texts: string[]): Promise<number[][]> {
        const response = await fetch(
            "https://integrate.api.nvidia.com/v1/embeddings",
            {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${this.apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    input: texts,
                    model: this.model,
                    input_type: "passage",
                }),
            }
        );

        const data = await response.json();

        return data.data.map((item: any) => item.embedding);
    }

    async embedQuery(text: string): Promise<number[]> {
        const response = await fetch(
            "https://integrate.api.nvidia.com/v1/embeddings",
            {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${this.apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    input: [text],
                    model: this.model,
                    input_type: "query",
                }),
            }
        );

        const data = await response.json();

        return data.data[0].embedding;
    }
}
// 1. Setup Embeddings (Local Ollama)
// const embeddings = new OllamaEmbeddings({
//     model: 'mxbai-embed-large',
//     baseUrl: 'http://localhost:11434'
// });
// 1. Switch from Ollama to NVIDIA Embeddings

//     const embeddings = new OpenAIEmbeddings({
//     apiKey: process.env.NVIDIA_LLAMA_EMB_API_KEY,
//     configuration: {
//         baseURL: "https://integrate.api.nvidia.com/v1",
//     },
//     model: "nvidia/llama-3.2-nv-embedqa-1b-v2",


// }as any);
const embeddings = new NVIDIAEmbeddings(process.env.NVIDIA_LLAMA_EMB_API_KEY!,
    "nvidia/llama-3.2-nv-embedqa-1b-v2",);



// 2. Load & Split (Only run if index doesn't exist to save time)
const VECTOR_STORE_PATH = "./nike_index";
let vectorStore: FaissStore;

if (fs.existsSync(VECTOR_STORE_PATH)) {
    console.log("Loading existing index... (Instant)");
    vectorStore = await FaissStore.load(VECTOR_STORE_PATH, embeddings);
} else {
    console.log("Creating new index... (Slow process)");
    const loader = new PDFLoader("./nke-10k-2023.pdf");
    const docs = await loader.load();
    const textSplitters = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });
    const allSplits = await textSplitters.splitDocuments(docs);
    // const allSplits = (await textSplitters.splitDocuments(docs)).slice(0, 10);

    vectorStore = await FaissStore.fromDocuments(allSplits, embeddings);
    await vectorStore.save(VECTOR_STORE_PATH);
}

// 3. Setup Gemma 4 (NVIDIA NIM)
const model = new ChatOpenAI({
    apiKey: process.env.NVIDIA_GEMMA_API_KEY, // Ensure this is in your .env
    configuration: {
        baseURL: "https://integrate.api.nvidia.com/v1",
    },
    model: "google/gemma-4-31b-it",
});

// 4. The RAG Process
// const query = "What was Nike's total revenue in 2023 and how does it compare to 2022?";
const query = "Based on the 10-K, what are the top 3 biggest risks to Nike's future growth, and how is management planning to mitigate them?";


// Retrieve chunks
const results = await vectorStore.similaritySearch(query, 6);
const context = results.map(res => res.pageContent).join("\n\n");

// Generate Answer
console.log("\n--- Consulting Gemma 4 (NVIDIA NIM) ---");

const response = await model.stream([
    ["system", `You are a financial analyst. Use the following Nike 10-K context to answer the user.
    If the answer isn't in the context, say you don't know.
    
    Context: ${context}`],
    ["human", query],
]);

console.log("\nFINAL ANSWER:");
for await (const chunk of response) {
    process.stdout.write(chunk.content.toString() || "");
}
console.log("\n\n--- End of Report ---");
