# # scripts/train_chatbot.py
# from llm.chatbot_assistant import ChatbotAssistant

# if __name__ == "__main__":
#     bot = ChatbotAssistant("llm/intents.json")
#     bot.parse_intents()
#     bot.prepare_data()
#     bot.train_model(batch_size=8, lr=1e-3, epochs=80)
#     bot.save_model("llm/chatbot_model.pth", "llm/dimensions.json")