# PyTorch Flask API

This repo contains a sample code to show how to create a Flask API server by deploying our PyTorch model. This is a sample code which goes with [tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html).

If you'd like to learn how to deploy to Heroku, then check [this repo](https://github.com/avinassh/pytorch-flask-api-heroku).


## How to 

Install the dependencies:

    pip install -r requirements.txt

From another tab, send the image file in a request:

    curl -X POST -F file=@cat_pic.jpeg http://localhost:5000/predict

or using HTTPie:
```sh
http -v POST 127.0.0.1:5000/predict --form file@./_static/shark.jpeg
```


## License

The mighty MIT license. Please check `LICENSE` for more details.
