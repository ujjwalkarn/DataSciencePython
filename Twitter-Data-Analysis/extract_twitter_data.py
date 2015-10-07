"""
Created on Sun Oct 04 23:10:41 2015
@author: ujjwal.karn
"""

#first, install pip by following instructions here: http://stackoverflow.com/questions/4750806/how-to-install-pip-on-windows 
#then, to install tweepy library, go to Anaconda command prompt and type: pip install tweepy
#once tweepy is installed, run the codes below:

import tweepy    #this will give an error if tweepy is not installed properly
from tweepy import OAuthHandler
 
#provide your access details below 
access_token = "xxxxxxxx"
access_token_secret = "xxxxxxxx"
consumer_key = "xxxxxxxx"
consumer_secret = "xxxxxxxx"
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
api = tweepy.API(auth)    
    
from tweepy import Stream
from tweepy.streaming import StreamListener
 
class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('C:\\Users\\ujjwal.karn\\Desktop\\Tweets\\python.json', 'a') as f:  #change location here
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
 
twitter_stream = Stream(auth, MyListener())

#change the keyword here
twitter_stream.filter(track=['#cricket'])
