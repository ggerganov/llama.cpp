#for x in list(globals()):
#    print("GLOBAL",x,globals()[x],"\n")
# any global variables set here will be available later as well!

#print("debug input:\n" + llm_input  + "\n")
#foobar ="test"
#if llm_state in ("params", statement, antiprompt,)

def entrypoint():
    global llm_output
    global llm_input
    global llm_state
    llm_output = llm_input
    if llm_state == "antiprompt":
        #used to check each token if you want to stop early
        return
    elif llm_state == "params":
        # first time it is called it returns the state via llm_output that will be used 
        return 
    elif llm_state == "statement":
        if "<GO>" in llm_input:
            llm_input = llm_input.replace("<GO>","")
            try:
                v= eval(llm_input)
                llm_output = "Check that the evaluation of```" + llm_input + "``` Produced:"+ str(v) + " STOP";
            except Exception as e:
                #print(e)
                llm_output = "generate a simple python expression to be evaluated. to evaluate your work emit the word <GO> and the python code will be evaluated.  Please correct the python error in Evaluation of ```" + llm_input + "``` Produced Output:"+ str(e) + "now consider the original task"+ llm_start + " STOP"
                
if __name__ == "__main__":
    entrypoint()
