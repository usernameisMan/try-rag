/**
 * 这是一个极简的 RAG (Retrieval-Augmented Generation，检索增强生成) 演示程序。
 * 
 * 核心概念科普：什么是 Embedding（嵌入）？
 * 1. 【理解 Embedding】：人类的文字（如“苹果”、“香蕉”）计算机是无法直接理解其“含义”的。Embedding 模型的作用就是
 *    将文字（可以是词、句子、甚至整段文章）转化为一个高维的“数字向量”（也就是一堆浮点数组成的数组，比如 1536 维的向量）。
 * 2. 【语义空间】：这些数字向量在多维空间中是有位置的。最神奇的是，**语义相近的词或句子，在空间中的位置（距离）也是接近的**。
 *    例如，“小狗”和“小猫”的向量距离会比“小狗”和“手机”的向量距离近得多。
 * 3. 【向量检索】：当我们输入一个问题（如“东东是谁？”）时，我们同样把问题转化为向量。然后在数据库中寻找
 *    和这个问题向量“距离最近”（即语义最相似）的文档片段。这就是“向量数据库检索”。
 * 4. 【检索增强生成】：检索出最相关的片段后，我们把这些片段作为“背景参考资料”和用户问题一起打包，发给大语言模型（LLM）。
 *    大语言模型通过阅读这些参考资料，就能给出准确且基于事实的回答，避免了“胡说八道”（幻觉）。
 */

import "dotenv/config"; // 1. 载入 .env 配置文件，将里面配置的 API_KEY 等敏感变量加载到 process.env 中，避免代码中直接硬编码 key。
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai"; // 2. 引入大模型组件：ChatOpenAI 用于生成回答；OpenAIEmbeddings 用于计算文本的向量（Embedding）。
import { Document } from "@langchain/core/documents"; // 3. 导入 Document 类：它是 LangChain 中表示文档的标准格式，包含文本内容 (pageContent) 和元数据 (metadata)。
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory"; // 4. 导入内存向量数据库：在内存中临时存储文档和它们对应的 Embedding 向量。

// 初始化大语言模型（LLM）
// 它的职责是：在最后一步，根据我们提供给它的“故事片段”（背景知识）和“问题”，进行分析并用温暖的人类语言写出最终的回答。
const model = new ChatOpenAI({
  temperature: 0, // 温度设置为 0，表示让模型的回答尽可能确定和严谨，减少随机性。
  model: process.env.MODEL_NAME, // 从环境变量读取使用的 LLM 模型名称（例如 gpt-4o 等）
  apiKey: process.env.OPENAI_API_KEY, // API 密钥
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL, // 允许通过代理或第三方平台（如 OneAPI）的 Base URL 访问
  },
});

// 初始化 Embedding 模型
// 它的职责是：把我们的文本（包括文档和后面的问题）转换成高维数字向量（Embedding），用于计算它们之间的语义相似度。
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.EMBEDDINGS_MODEL_NAME, // 使用的嵌入模型名称（例如 text-embedding-3-small）
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

// 定义我们的本地知识库文档（这里是一个关于“光光和东东”的小故事）
// 使用 Document 包装每一段内容，可以附带 metadata（元数据，如章节、角色等），这在复杂搜索、过滤和排序时非常有用。
const documents = [
  new Document({
    pageContent: `光光是一个活泼开朗的小男孩，他有一双明亮的大眼睛，总是带着灿烂的笑容。光光最喜欢的事情就是和朋友们一起玩耍，他特别擅长踢足球，每次在球场上奔跑时，就像一道阳光一样充满活力。`,
    metadata: {
      chapter: 1,
      character: "光光",
      type: "角色介绍",
      mood: "活泼",
    },
  }),
  new Document({
    pageContent: `东东是光光最好的朋友，他是一个安静而聪明的男孩。东东喜欢读书和画画，他的画总是充满了想象力。虽然性格不同，但东东和光光从幼儿园就认识了，他们一起度过了无数个快乐的时光。`,
    metadata: {
      chapter: 2,
      character: "东东",
      type: "角色介绍",
      mood: "温馨",
    },
  }),
  new Document({
    pageContent: `有一天，学校要举办一场足球比赛，光光非常兴奋，他邀请东东一起参加。但是东东从来没有踢过足球，他担心自己会拖累光光。光光看出了东东的担忧，他拍着东东的肩膀说："没关系，我们一起练习，我相信你一定能行的！"`,
    metadata: {
      chapter: 3,
      character: "光光和东东",
      type: "友情情节",
      mood: "鼓励",
    },
  }),
  new Document({
    pageContent: `接下来的日子里，光光每天放学后都会教东东踢足球。光光耐心地教东东如何控球、传球和射门，而东东虽然一开始总是踢不好，但他从不放弃。东东也用自己的方式回报光光，他画了一幅画送给光光，画上是两个小男孩在球场上一起踢球的场景。`,
    metadata: {
      chapter: 4,
      character: "光光和东东",
      type: "友情情节",
      mood: "互助",
    },
  }),
  new Document({
    pageContent: `比赛那天终于到了，光光和东东一起站在球场上。虽然东东的技术还不够熟练，但他非常努力，而且他用自己的观察力帮助光光找到了对手的弱点。在关键时刻，东东传出了一个漂亮的球，光光接球后射门得分！他们赢得了比赛，更重要的是，他们的友谊变得更加深厚了。`,
    metadata: {
      chapter: 5,
      character: "光光和东东",
      type: "高潮转折",
      mood: "激动",
    },
  }),
  new Document({
    pageContent: `从那以后，光光和东东成为了学校里最要好的朋友。光光教东东运动，东东教光光画画，他们互相学习，共同成长。每当有人问起他们的友谊，他们总是笑着说："真正的朋友就是互相帮助，一起变得更好的人！"`,
    metadata: {
      chapter: 6,
      character: "光光和东东",
      type: "结局",
      mood: "欢乐",
    },
  }),
  new Document({
    pageContent: `多年后，光光成为了一名职业足球运动员，而东东成为了一名优秀的插画师。虽然他们走上了不同的道路，但他们的友谊从未改变。东东为光光设计了球衣上的图案，光光在每场比赛后都会给东东打电话分享喜悦。他们证明了，真正的友情可以跨越时间和距离，永远闪闪发光。`,
    metadata: {
      chapter: 7,
      character: "光光和东东",
      type: "尾声",
      mood: "温馨",
    },
  }),
];

// 创建向量数据库（Vector Store）
// MemoryVectorStore.fromDocuments 的执行过程：
// 1. 底层会自动调用上面定义的 `embeddings` 模型，把这 7 个 Document 的 `pageContent` 文字发给 API 换取对应的 7 个高维 Embedding 向量。
// 2. 然后将这些向量和原本的 Document 关联，并保存在内存中。
const vectorStore = await MemoryVectorStore.fromDocuments(
  documents,
  embeddings,
);

// 将向量数据库转换成一个“检索器 (Retriever)”
// `asRetriever({ k: 3 })` 表示：检索器每次接收到查询（Question）时，会返回最相关的 3 个文档（即 k = 3）。
const retriever = vectorStore.asRetriever({ k: 3 });

// 待提问的问题
const questions = ["东东和光光是怎么成为朋友的?"];

for (const question of questions) {
  console.log("=".repeat(80));
  console.log(`问题: ${question}`);
  console.log("=".repeat(80));

  // 步骤 1：利用检索器（Retriever）获取最相关的文档。
  // 执行过程：
  // 1. 检索器会自动将我们的问题 `"东东和光光是怎么成为朋友的?"` 通过 embeddings 模型转化为一个临时的“问题向量”。
  // 2. 在向量数据库中计算这个“问题向量”与库里所有“文档向量”的余弦相似度（或者其他距离算法）。
  // 3. 返回得分最高的 k (3) 个文档。
  const retrievedDocs = await retriever.invoke(question);

  // 步骤 2：直接从向量库中进行相似度搜索，并获取数值评分（Score）。
  // 这个操作与上面类似，但它会连同评分一起返回，用来让我们直观看到匹配的程度有多高。
  //在这行代码中，数字 3 代表 要返回的最相似文档的数量（在机器学习和检索中通常被称为参数 k）。
  const scoredResults = await vectorStore.similaritySearchWithScore(
    question,
    3,
  );

  // 打印用到的文档和相似度评分
  console.log("\n【检索到的文档及相似度评分】");
  retrievedDocs.forEach((doc, i) => {
    // 找到当前文档在 scoredResults 中对应的元组
    // scoredResults 格式为：[[Document, score], [Document, score], ...]
    const scoredResult = scoredResults.find(([scoredDoc]) =>
      scoredDoc.pageContent === doc.pageContent
    );
    const score = scoredResult ? scoredResult[1] : null;

    // 注意：这里的 score 通常代表“距离”（如余弦距离或 L2 距离）。
    // 在很多向量库的实现中，距离越小（越接近 0），说明它们在语义空间中越贴近，也就是相似度越高。
    // 因此，用 `1 - score` 可以把“距离”转换为一个直观的“相似度评分”（0 到 1 之间，越接近 1 说明越相似）。
    const similarity = score !== null ? (1 - score).toFixed(4) : "N/A";

    console.log(`\n[文档 ${i + 1}] 相似度: ${similarity}`);
    console.log(`内容: ${doc.pageContent}`);
    console.log(`元数据: 章节=${doc.metadata.chapter}, 角色=${doc.metadata.character}, 类型=${doc.metadata.type}, 心情=${doc.metadata.mood}`);
  });

  // 步骤 3：构建 RAG 提示词（Prompt）。
  // 我们要把刚才查出来的 3 个最相关的“片段”组合成一个“参考资料/上下文（Context）”。
  const context = retrievedDocs
    .map((doc, i) => `[片段${i + 1}]\n${doc.pageContent}`)
    .join("\n\n━━━━━\n\n");

  // 将检索到的“背景上下文（context）”和“用户问题（question）”填入我们设计好的模版中。
  // 我们显式地告诉大模型：“基于以下故事片段回答问题”，并加上一些语气指令（“温暖生动的语言”），限制它的回答范围。
  const prompt = `你是一个讲友情故事的老师。基于以下故事片段回答问题，用温暖生动的语言。如果故事中没有提到，就说"这个故事里还没有提到这个细节"。

故事片段:
${context}

问题: ${question}

老师的回答:`;

  console.log("\n【AI 回答】");
  // 步骤 4：调用大语言模型（LLM），将构建好的 Prompt 发送给它，生成最终的回答。
  const response = await model.invoke(prompt);
  console.log(response.content);
  console.log("\n");
}
