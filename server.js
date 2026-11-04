import express from "express";
import { createDecartClient } from "@decartai/sdk";
import { fileURLToPath } from "node:url";
import path from "node:path";

const app = express();
const port = Number(process.env.PORT) || Number(process.argv[2]) || 3000;
const publicDirectory = path.join(path.dirname(fileURLToPath(import.meta.url)), "public");
const decart = process.env.DECART_API_KEY ? createDecartClient({ apiKey: process.env.DECART_API_KEY }) : null;

app.disable("x-powered-by");
app.use(express.static(publicDirectory));

// The browser gets a short-lived key per session; the permanent key never leaves this server.
app.post("/api/realtime-token", async (_request, response) => {
  if (!decart) return response.status(503).json({ error: "DECART_API_KEY غير موجود في ملف .env" });
  try {
    const { apiKey } = await decart.tokens.create({ expiresIn: 300 });
    response.set("Cache-Control", "no-store").json({ apiKey });
  } catch (error) {
    console.error("token creation failed", error);
    response.status(502).json({ error: "تعذّر الاتصال بخدمة الذكاء الاصطناعي" });
  }
});

app.listen(port, () => {
  console.log(`Portal ready on http://localhost:${port}`);
  if (!decart) console.log("DECART_API_KEY is missing from .env — the portal will open but the AI stays off.");
});
