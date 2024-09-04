from flask import Flask, render_template, request, redirect, url_for
from main_sentiment_analysis2 import predict_sentiment
import webbrowser
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index1():
    prediction = None
    user_input = ""

    if request.method == 'POST':
        text = request.form['text']
        user_input = text

        chrome_driver_path = "C:/Users/Kapihll kumar/Downloads/chromedriver-win64/chromedriver.exe"
        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_driver_path))

        search_url = f"https://www.google.com/search?q={text}+imdb"
        
        webbrowser.get('chrome').open(search_url)
        
        time.sleep(5)

        driver = webdriver.Chrome()

        driver.get(search_url)

        time.sleep(5)  

        search_results = driver.find_elements(By.CSS_SELECTOR, "div.g")
        for result in search_results:
            link = result.find_element(By.CSS_SELECTOR, "a")
            if "imdb.com/title/" in link.get_attribute("href"):
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.g a"))).click()
                break

        time.sleep(5)

        try:
            reviews_link = driver.find_element(By.CLASS_NAME, "ipc-lockup-overlay")
            reviews_link.click()
            time.sleep(2) 
        except Exception as e:
            print("Error clicking on 'Reviews' link:", e)
        
        try:
            image_element = driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/div[4]/img')
            image_src = image_element.get_attribute("src")
            if "imdb-logo" not in image_src: 
                image_response = requests.get(image_src)
                with open("static/images/background.jpg", "wb") as file:
                    file.write(image_response.content)
        except Exception as e:
            print("Error downloading image:", e)

        driver.back()

        reviews_url = driver.current_url + "reviews"


        driver.get(reviews_url)

        time.sleep(3) 

        for _ in range(3): 
            try:
                load_more_button = driver.find_element(By.CLASS_NAME, "ipl-load-more__button")
                load_more_button.click()
                time.sleep(3) 
            except Exception as e:
                print("Error clicking on 'Load More Reviews' button:", e)
                break

        user_reviews = driver.find_elements(By.CLASS_NAME, "text.show-more__control")

        with open("reviews.txt", "w", encoding="utf-8") as file:
            for review in user_reviews:
                file.write(review.text + "\n")

        print("User reviews saved to 'reviews.txt'")


        driver.quit()

        with open("reviews.txt", "r", encoding="utf-8") as file:
            reviews = file.read()
            prediction = predict_sentiment(reviews)

        return render_template('output.html', prediction=prediction)

    return render_template('index1.html', user_input=user_input)

@app.route('/output')
def output():
    return redirect(url_for('index1'))

if __name__ == '__main__':
    app.run(debug=True)
