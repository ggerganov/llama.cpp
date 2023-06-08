import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

const chat = [
  { human: "Hello, Assistant.",
    assistant: "Hello. How may I help you today?" },
  { human: "Please tell me the largest city in Europe.",
    assistant: "Sure. The largest city in Europe is Moscow, the capital of Russia." },
]

function format_prompt(question) {
  return "A chat between a curious human and an artificial intelligence assistant. "
    + "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
    + chat.map(m => `### Human: ${m.human}\n### Assistant: ${m.assistant}`).join("\n")
    + `\n### Human: ${question}\n### Assistant:`
}

async function ChatCompletion(question) {
  const result = await fetch("http://127.0.0.1:8080/completion", {
    method: 'POST',
    body: JSON.stringify({
      prompt: format_prompt(question),
      temperature: 0.2,
      top_k: 40,
      top_p: 0.9,
      n_keep: 29,
      n_predict: 256,
      stop: ["\n### Human:"], // stop completion after generating this
      stream: true,
    })
  })

  if (!result.ok) {
    return;
  }

  let answer = ''

  for await (var chunk of result.body) {
    const t = Buffer.from(chunk).toString('utf8')
    if (t.startsWith('data: ')) {
      const message = JSON.parse(t.substring(6))
      answer += message.content
      process.stdout.write(message.content)
      if (message.stop) break;
    }
  }

  process.stdout.write('\n')
  chat.push({ human: question, assistant: answer })
}

const rl = readline.createInterface({ input, output });

while(true) {

  const question = await rl.question('> ')
  await ChatCompletion(question);

}

