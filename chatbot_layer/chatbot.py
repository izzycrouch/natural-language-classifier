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
    

    def generate_reply(self, is_extract: bool, prompt: str):

        new_prompt = '<|user|>\n' + prompt + '<|end|>\n'
        

        if is_extract:
            system_prompt = """
                         <|system|>
                        You are an expert headline extraction assistant. 
                        Your task is to identify and extract the news headline from the user input which is enclosed in quotation marks. 
                        Return only the extracted headline, do not add anything else to the response.
                        If no headline is found, respond with "None".<|end|>
                        """
            
                
            # examples = """ Here are some examples of inputs and how you should respond:
            #         <|user|>
            #         What are the highlights of 'Spain and Germany renew battle in Nations League final showdown'?<|end|>
            #         <|assistant|>
            #         Spain and Germany renew battle in Nations League final showdown<|end|>
            #         <|user|>
            #         I got sent this article: Government to ditch day-one unfair dismissal policy from workers. What are the talking points from this article?<|end|>
            #         <|assistant|>
            #         Government to ditch day-one unfair dismissal policy from workers<|end|>
            #         <|user|>
            #         What do you think about oranges?<|end|>
            #         <|assistant|>
            #         No title detected<|end|>
            #         Here is the actual user input:
            #         <|user|>
            #         """
            
            # input = system_prompt + examples + new_prompt
            input = system_prompt + new_prompt
            encoded_input = self.encode_prompt(input)
            input_ids = encoded_input['input_ids']
            attention_mask = torch.ones_like(input_ids)

            generate = self.model.generate(input_ids=input_ids, attention_mask=attention_mask , pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=50, do_sample=True, temperature=0.01)
        
        else:
            system_prompt = """
                    <|system|>
                    Given an article title and a classifier-predicted category, generate a friendly message for the user. 
                    The message must:
                    - Clearly state the extracted article title.
                    - Inform the user of the predicted category.
                    - Provide a short explanation of why the article might fit that category.
                    - Address the user directly and keep the tone informative and concise.
                    """

            if self.chat_history_ids == None:
                input = system_prompt + new_prompt
                encoded_input = self.encode_prompt(input)
                input_ids = encoded_input['input_ids']
                attention_mask = torch.ones_like(input_ids)
            
            else:
                encoded_prompt = self.encode_prompt(new_prompt)
                prompt_ids = encoded_prompt['input_ids']
                
                input_ids = torch.cat((self.chat_history_ids, prompt_ids), dim=1).to(self.device)
                attention_mask = torch.ones_like(input_ids).to(self.device)
            
            generate = self.model.generategenerate = self.model.generate(input_ids=input_ids, attention_mask=attention_mask , pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=200, temperature=0.7, top_p=0.95, top_k=50, do_sample=True)

            self.chat_history_ids = generate

        token_ids = generate[0].tolist()
        list_input_ids = input_ids[0].tolist()
        input_len = len(list_input_ids)
        reply_ids = token_ids[input_len:]
        decoded_reply = self.decode_reply(reply_ids)

        stripped_reply = decoded_reply.strip()
        
        if stripped_reply.startswith('<|assistant|>'):
            stripped_reply = stripped_reply.replace('<|assistant|>', '')

        if '<|end|>' in stripped_reply:
            stripped_reply = stripped_reply.replace('<|end|>', '')
        
        reply = stripped_reply.strip()

        return reply
    
    
    def reset_history(self):
        self.chat_history_ids = None
        return self.chat_history_ids

    