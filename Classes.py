import pandas as pd

class PatentExtract():

    def __init__(self, node):
        self.node = node

    def getAttribute(self,attr):
        attribute = []
        attribute.append(self.node.getAttribute(f"{attr}"))
        
        return attribute

    def singleTagTextExtract(self,tag):
        text_NODE = self.node.getElementsByTagName(f"{tag}")
        text = []

        for i in text_NODE:
            text.append(i.firstChild.data)
        
        return text
        
    def multiTagTextExtract(self,tag,inner_tag,attr=None):
        text_NODE = self.node.getElementsByTagName(f"{tag}")
        text = []
        attr = []
        attr_flag = False

        for i in text_NODE:
            if attr != None:
                attr.append(i.getAttribute(f'{attr}'))
                attr_flag = True

            for j in i.getElementsByTagName(f'{inner_tag}'):
                text.append(j.firstChild.data)

        if attr_flag == True:
            return text, attr
        else:
            return text
        

class TableCreator():

    def __init__(self):
        self.procedure_lang = []
        self.court_type = []
        self.appeal_num = []
        self.fact_lang = []
        self.summary_facts = []
        self.decision_reason = []
        self.order = []
        self.date = []
        self.ecli = []
        self.title = []
        self.board_code = []
        self.keywords = []

    def create_table(self):
        
        dictionary = {'Procedure Language':self.procedure_lang,
                      'Court Type':self.court_type,
                      'Appeal Number':self.appeal_num,
                      'Document Language':self.fact_lang,
                      'Summary Facts':self.summary_facts,
                      'Decision Reasons':self.decision_reason,
                      'Order':self.order,
                      'Date':self.date,
                      'ECLI':self.ecli,
                      'Invention Title':self.title,
                      'Board Code':self.board_code,
                      'Keywords':self.keywords}
        
        self.df = pd.DataFrame(dictionary) 

    def append(self,name,info):
        name.append(info)
