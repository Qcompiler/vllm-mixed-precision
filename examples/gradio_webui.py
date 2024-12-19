import gradio as gr
import requests
import json

MAX_HISTORY_LEN=5000

def chat_streaming(query,history):
    # 调用api_server
    response=requests.post('http://localhost:8000/chat',json={
        'query':query,
        'stream': True,
        'history':history
    },stream=True)
    
    # 流式读取http response body, 按\0分割
    for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
        if chunk:
            data=json.loads(chunk.decode('utf-8'))
            text=data["text"].rstrip('\r\n') # 确保末尾无换行
            yield text


PLACEHOLDER = """

<head>
    <script type="text/javascript">
        function send() {
            var chatBox = document.getElementById('chat-box');
             
            chatBox.scrollTop = chatBox.scrollHeight;   
        } 
    </script>
</head>


<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">AI Lawyer</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything ...</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

scroll_chat_js = """
    function scrollChat() {
        var element = document.getElementById('chatbot-clinic-chat');
        element.scrollTop = element.scrollHeight;
    }
"""
with gr.Blocks(css='.qwen-logo img {height:100px; width:600px; margin:0 auto;}') as app:
    with gr.Row():
        logo_img=gr.Image('lenovo.jpg',elem_classes='qwen-logo')
    with gr.Row():
        chatbot=gr.Chatbot(label='Lenovo LLM demo',height=800,   elem_id="chatbot-clinic-chat")
        #chatbot.change(None, js=scroll_chat_js, show_progress="hidden")
    with gr.Row():
        query_box=gr.Textbox(label='提问',autofocus=True,lines=5, autoscroll = True)
    with gr.Row():
        clear_btn=gr.ClearButton([query_box,chatbot],value='清空历史')
        submit_btn=gr.Button(value='提交')

    def chat(query,history):
        for response in chat_streaming(query,history):
            yield '',history+[(query,response)]
        history.append((query,response))
        while len(history) > MAX_HISTORY_LEN:
            history.pop(0)
    
    # 提交query
    submit_btn.click(chat,[query_box,chatbot],[query_box,chatbot])
    # query_box.submit(chat,[query_box,chatbot],[query_box,chatbot])
    chatbot.change(None, js=scroll_chat_js)
if __name__ == "__main__":
    app.queue(200)  # 请求队列
    app.launch(server_name='0.0.0.0',max_threads=500) # 线程池