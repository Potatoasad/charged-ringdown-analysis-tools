with open("GW150914.ini") as f:
    lines = f.readlines()

event_list = ["GW200129_065458", 
"GW200224_222234",
"GW200311_115853",
"GW150914",
"GW190519_153544",
"GW190521_074359",
"GW190706_222641",
"GW190910_112807",
"GW190521",
"GW170104",
"GW170823",
"GW190630_185205",
"GW190828_063405"]


for event_new in event_list:
	with open(event_new+".ini", 'w') as f:
		new_script = ''.join(lines)
		new_script = new_script.replace("GW150914", event_new)
		f.write(new_script)
