import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

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