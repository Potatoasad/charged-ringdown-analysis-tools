import ringdown

from ringdb import Database

db = Database('../Data')
db.initialize()

event = db.event("GW150914")

posts = event.posteriors()



