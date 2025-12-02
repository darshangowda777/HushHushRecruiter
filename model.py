# export the model
import pickle
import supervised  

if __name__ == "__main__":
    model = supervised.train()  
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved model to model.pkl")
