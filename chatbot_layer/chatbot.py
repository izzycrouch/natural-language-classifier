from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:

    def __init__(self):
        model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
        self.chat_history_ids = None

    def encode_prompt(self, prompt: str):
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return encoded
    
    def decode_reply(self, reply_ids: list[int]):
        decoded = self.tokenizer.decode(reply_ids, skip_special_tokens=True)
        return decoded
    
    def extract_title(self, prompt: str):

        system_prompt = """
                    <|system|>
                    Role: You are an expert data extraction assistant. 
                    Task: You perfectly extract and return only the news headline from the input text, if none are detected you return No title detected.
                    Example:
                    input: my friend sent me this article 'The US government ruins Thanksgiving: it's a South Park holiday special' should i read it?
                    response: The US government ruins Thanksgiving: it's a South Park holiday special
                    """

        examples = """
                    <|user|>
                    What are the highlights of 'Spain and Germany renew battle in Nations League final showdown'?<|end|>
                    <|assistant|>
                    Spain and Germany renew battle in Nations League final showdown<|end|>
                    <|user|>
                    I got sent this article: Government to ditch day-one unfair dismissal policy from workers. What are the talking points from this article?<|end|>
                    <|assistant|>
                    Government to ditch day-one unfair dismissal policy from workers<|end|>
                    <|user|>
                    Should i watch the new season of stranger things based on this article 'Stranger Things season five review - this luxurious final run will have you standing on a chair, yelling with joy'?<|end|>
                    <|assistant|>
                    Stranger Things season five review - this luxurious final run will have you standing on a chair, yelling with joy<|end|>
                    <|user|>
                    What day is it today?<|end|>
                    <|assistant|>
                    No title detected<|end|>
                    """

        new_prompt = '<|user|>\n' + prompt + '<|end|>\n'

        input = new_prompt + system_prompt + examples 
        encoded_input = self.encode_prompt(input)

        input_ids = encoded_input['input_ids']
        attention_mask = torch.ones_like(input_ids)
        
        generate = self.model.generate(input_ids=input_ids, attention_mask=attention_mask , pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=50, do_sample=False)

        token_ids = generate[0].tolist()
        list_input_ids = input_ids[0].tolist()
        input_len = len(list_input_ids)
        reply_ids = token_ids[input_len:]
        decoded_reply = self.decode_reply(reply_ids)

        stripped_reply = decoded_reply.strip()
        
        if stripped_reply.startswith('<|assistant|>'):
            stripped_reply = stripped_reply.replace('<|assistant|>', '')
        
        title = stripped_reply
        
        return title
    
    def generate_reply(self, prompt: str):
        new_prompt = '<|user|>/n' + prompt + '<|end|>'
        encoded_prompt = self.encode_prompt(new_prompt)

        prompt_ids = encoded_prompt['input_ids']
        
        system_prompt = "<|system|>\nYou are a friendly assistant who explains the results of a classification in natural language.<|end|>\n"

        if self.chat_history_ids == None:
            encoded_system_prompt = self.encode_prompt(system_prompt)
            system_prompt_ids = encoded_system_prompt['input_ids']

            input_ids = torch.cat((system_prompt_ids, prompt_ids), dim=1)
            attention_mask = torch.ones_like(input_ids)
        
        else:
            input_ids = torch.cat((self.chat_history_ids, prompt_ids), dim=1).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
        
        generate = self.model.generate(input_ids=input_ids, attention_mask=attention_mask , pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=200, temperature=0.4, top_p=0.95, top_k=50, do_sample=True)

        token_ids = generate[0].tolist()
        list_input_ids = input_ids[0].tolist()
        input_len = len(list_input_ids)
        reply_ids = token_ids[input_len:]
        decoded_reply = self.decode_reply(reply_ids)
        
        self.chat_history_ids = generate
    
        return decoded_reply

    def reset_history(self):
        self.chat_history_ids = None
        return self.chat_history_ids

