# Image classification of cats:

This project was made as a modified version of the final project in my [Coursera NN Certification](https://coursera.org/share/0d4da214b5a57844a18a312e7cdb3f17). The project is a Neural Network written in python from scratch. We can specify the hyperparameters for the neural network dynamically. The neural network will classify pictures as cats or non-cats.

### Running the script :

1. Find any images that you want to test with and store it in ```./src/training/images``` folder
2. Run ```./src/training/train.py``` with the following argument : ```--image <your_image_name_with_extention>```

**Example :** 
``` 
python train.py --image another_cat.jpg
```


### To Do :
1. code cleaning & restructuring ✔
2. add more hyperparameter options
3. implement Flask API 
4. make a Dash front-end
5. Containerize with Docker and deploy in AWS.

![cat_image](https://github.com/abhi094/Educational-Projects/blob/master/Neural%20Networks%20in%20Python/images/cat.png)
