import "dotenv/config";
import "cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { getEncoding } from "js-tiktoken";

const logDocument = new Document({
  pageContent: `[2024-01-15 10:00:00] INFO: Application started
[2024-01-15 10:00:05] DEBUG: Loading configuration file
[2024-01-15 10:00:10] INFO: Database connection established
[2024-01-15 10:00:15] WARNING: Rate limit approaching
[2024-01-15 10:00:20] ERROR: Failed to process request
[2024-01-15 10:00:25] INFO: Retrying operation
[2024-01-15 10:00:30] SUCCESS: Operation completed
[2026-01-10 14:30:00] INFO: 系统开始执行大规模数据迁移任务，本次迁移涉及核心业务数据库中的用户表、订单表、商品库存表、物流信息表、支付记录表、评论数据表等共计十二个关键业务表，预计处理数据量约500万条记录，数据总大小预估为280GB，迁移过程将采用分批次增量更新策略以减少对生产环境的影响，同时启用双写机制确保数据一致性，任务预计总耗时约3小时15分钟，迁移完成后将自动触发全面的数据一致性校验流程以及性能基准测试，请相关运维人员和DBA团队密切关注系统资源使用情况、网络带宽占用率以及任务执行进度，如遇异常情况请立即启动应急预案并通知技术负责人
`,
});

const enc = getEncoding("cl100k_base");

const logTextSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 150,
  chunkOverlap: 20,
  separators: ["\n", "。", "，"],
  lengthFunction: (text) => enc.encode(text).length,
});
const splitDocuments = await logTextSplitter.splitDocuments([logDocument]);

// console.log(splitDocuments);

splitDocuments.forEach((document) => {
  console.log(document);
  console.log("charater length:", document.pageContent.length);
  console.log("token length:", enc.encode(document.pageContent).length);
});
// RecursiveCharacterTextSplitter + lengthFunction 的情况
//当设置了lengthfunction chunk自动变为了计算token长度,
//然后当分割了之后就会计算分割的这段有没有超过chunk,没有就正常是一段,
//如果超过了就再进行拆分小于chunksize 并滑动衔接

/** 
这三种文本分割器是 LangChain 中最核心的工具。以下是对它们的简明对比和总结：


1. CharacterTextSplitter（单字符分割器）
工作原理：只使用一个指定的分隔符（默认是换行符 \n）将文本切开，然后拼接成块。
衡量单位：字符数。
优缺点：
优点：极其简单、直观。
缺点：过于死板。如果切分后的某一段本身就超过了 chunkSize，由于没有后备分隔符，它就无法再切，导致最终生成的 Chunk 依然会超标。
适用场景：格式非常工整、每段长度都很均匀的文档。


2. RecursiveCharacterTextSplitter（递归字符分割器）🌟 最推荐
工作原理：使用一个分隔符列表（默认是 ["\n\n", "\n", " ", ""]）从大到小递归切分。如果大分隔符切出来的块超标，就用下一个更细的分隔符继续切。
衡量单位：默认是字符数，但通过重写 lengthFunction 可以完美支持 Token 数。
优缺点：
优点：最智能。它会优先保证语义的完整性，尽量把段落、句子放在同一个 Chunk 里，避免在词语中间被拦腰截断。
缺点：计算稍微复杂一点点。
适用场景：绝大多数 RAG（知识库）应用、普通文章、PDF 文档等。


3. TokenTextSplitter（Token 分割器）
工作原理：不理会任何标点符号和语义。直接把整篇文本全部转换成一串 Token ID，然后像切香肠一样，强行每隔 chunkSize 个 Token 切刀。
衡量单位：只能是 Token 数。
优缺点：
优点：Token 数量绝对精准。每个块的大小分毫不差。
缺点：完全不讲语义。一句话可能在中间（甚至一个英文单词在字母中间）被硬生生切断，对大模型理解上下文非常不友好。
适用场景：对语义要求不高，但对大模型输入 Token 限制极其严苛的场景。

📊 总结对比表
分割器名称	核心特点	衡量单位	语义保留效果	推荐指数
CharacterTextSplitter	单个分隔符一刀切	字符数	一般	★★☆☆☆
RecursiveCharacterTextSplitter	多级分隔符递归切分	字符数（可自定义为 Token）	极好（优先保留段落/句子）	★★★★★ (首选)
TokenTextSplitter	纯按 Token 数量硬切	Token数	较差（会切断单词/句子）	★★★☆☆
*/
