import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";

// const document=[
//     new Document({
//         pageContent:'DOgs are great companions, known for their loyalty and friendliness',
//         metadata:{source:'mamals-pets-doc'}
//     }),
//     new Document({
//         pageContent:"Cats are independent pets that often enjoy their own space",
//         metadata:{source:'mamals-pets-doc'}
//     }),
// ];

const loader=new PDFLoader("./nke-10k-2023.pdf");
const docs=await loader.load();
console.log(docs.length);
console.log(docs[0].pageContent,docs[0].metadata);
const textSplitters=new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:200,
});
const allSplits=await textSplitters.splitDocuments(docs);
console.log(allSplits.length);
