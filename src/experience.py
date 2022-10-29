from datetime import datetime
import os

def create_experiments():
  today = datetime.now()

  if today.hour < 12:
    h = "00"
  else:
    h = "12"
  path= "./experiments/" + today.strftime('%Y%m%d')+ h
  if not os.path.exists(path):
    os.makedirs(path)
    print("Create experiments Dir : {}!".format(path))
  return path