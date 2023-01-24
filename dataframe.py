
import pandas as pd

user_details={'Name':['arun','mahesh'],"Amount":[1000,1000],"mail_id":['joviyalarun07@gmail.com','mahesh@gmail.com'],'card_number':[7,25]}

data_frame=pd.DataFrame(user_details)

data_frame.to_csv('details.csv')
