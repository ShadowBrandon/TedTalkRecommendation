from flask import Flask, request, render_template
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


with open('cleaned_talks.pkl', 'rb') as picklefile:
    cleaned_talks = pickle.load(picklefile)

with open('ted_all_trimmed.pkl', 'rb') as picklefile:
    ted_all = pickle.load(picklefile)

with open('lda.pkl', 'rb') as picklefile:
    lda = pickle.load(picklefile)

with open('lda_data.pkl', 'rb') as picklefile:
    lda_data = pickle.load(picklefile)

with open('vectorizer.pkl', 'rb') as picklefile:
    vectorizer = pickle.load(picklefile)

with open('topic_names.pkl', 'rb') as picklefile:
    topic_names= pickle.load(picklefile)

titles = ted_all['title']


#---------- MODEL IN MEMORY ----------------#
def get_recommendations(first_article, model, vectorizer, training_vectors,title, ind):
    #print("First")
    rec_dict = defaultdict(list)

    new_vec = model.transform(vectorizer.transform([first_article]))
    nn = NearestNeighbors(n_neighbors=4, metric='cosine', algorithm='brute')
    nn.fit(training_vectors)
    results = nn.kneighbors(new_vec)

    #print(results) #list(array(of distance), array(of show numbers)
    #print('\n')

    recommend_list = results[1][0] #List of show numbers
    #print(recommend_list) #List of show numbers 

    scores = results[0] #list of distance
    #print(scores)#list of distance

     #will hold the recomendation url
    ss = np.array(scores).flat ##
    print (ss)
    print('\n')


    for i, resp in enumerate(recommend_list): 
    #for every item in the list of show-numbers
        rec_dict[i] = [resp, ted_all.iloc[resp,-4],ted_all.iloc[resp,-3],topic_names.iloc[resp,0],ted_all.iloc[resp,1],ss[i]]
        #solution.append(resp, ted_all.iloc[resp,-4],ted_all.iloc[resp,-3],topic_names.iloc[resp,0],ted_all.iloc[resp,1],ss[i])

        print('\n--- ID ---\n', + resp)
        print('--- Talk Show ---')
        print(ted_all.iloc[resp,-4])
        print('--- Tags ---')
        print(ted_all.iloc[resp,-3])
        print('--- Topic ---')
        print(topic_names.iloc[resp,0])
        print('--- URL ---')
        print(ted_all.iloc[resp,1])
        print('--- Distance ---\n', + ss[i])
        print('\n')
        print(rec_dict)
        # rec_dict[i] = {'':}
        # #print('--- ID ---\n', + resp) 
        # #print('--- dist ---\n', + ss[i])
        # print('\n')
        
        # print('\n')
        # print(ted_all.iloc[resp,1]) #this is a link to the url
        # print('\n')
        # print('\n')
        # print('During For-Loop')
        # print(rec_dict[i] = append(ted_all.iloc[resp,1])) #attempts to append the


    # print("After-for-loop")
    # print("rec_dict", rec_dict)
    # rec_dict["0"].append(title)
    # print(rec_dict[0])
    # print("rec_dict[0]", rec_dict["0"])
    # print("Crashes Here HEREE :(")
    # rec_dict["0"].append(topic_names.iloc[ind,0])
    # print("Crash???")
    # print (rec_dict)
    # print("Crash?????????")
    return rec_dict

    


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    return render_template('index.html')
    # """
    # Homepage: serve our visualization page, ted_rec.html
    # """
    # with open("ted_rec.html", 'r') as viz_file:
    #     return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/ted", methods=["POST", "GET"])
def ted():

    user_ted_response = request.form['TedTalkTitle']
    print(user_ted_response)
    #print(lda)
    #print(titles)
    found_match = 0

    for talk_ind, element in enumerate(titles): #I can't find user_ted_response as part of the column of title,
        print(element) #title
        print(talk_ind) #index
        if element == user_ted_response:
            print("IT IS A TITLE")
            recs = get_recommendations(cleaned_talks[int(talk_ind)],lda, vectorizer, lda_data, titles, talk_ind)
            print("Exit: get_rec given a TITLE")
            found_match = 1

    if (found_match == 0): #if we didn't find then it is a number
        print("IT IS A NUMBER")
        #print(cleaned_talks) It is a [talk1,talk2,talk3..]
        #print(type(cleaned_talks))
        #print(int(user_ted_response)

        #print("Enter: get_rec given ID")
        recs = get_recommendations(cleaned_talks[int(user_ted_response)],lda, vectorizer, lda_data, titles, user_ted_response)
        #print("Exit: get_rec given ID ")
        #final_message = muffin_or_cupcake(amounts)
        #return render_template('index.html', message=final_message)

    #print(recs.values())
    # Put the result in a nice dict so we can send it as json
    #results = {"recommends": recs[1]}
    #print(results)
    #recs['0']
    return render_template('index.html', message = (recs)) #delete['0'] #best case rec_dict[0] just links

    #return flask.json.dump(recs,fp)
#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
if __name__ == '__main__':
    app.run(debug=True)

# app.run(host='0.0.0.0')
# app.run(debug=True)