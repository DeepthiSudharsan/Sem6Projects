# Importing libraries
import re
import sys
import streamlit as st
from io import StringIO
import time
# Some basic beautification
st.title('HTML to Text')
st.subheader('Convert HTML to plain text')
# Adding an image from Unsplash to the side bar 
st.sidebar.image("https://images.unsplash.com/photo-1518773553398-650c184e0bb3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80", width=None)
st.sidebar.markdown("Photo by Pankaj Patel on Unsplash")
# Github Repository where the code has been hosted
st.sidebar.subheader("Github : ")
st.sidebar.markdown("[![Github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJGtP-Pq0P67Ptyv3tB7Zn2ZYPIT-lPGI7AA&usqp=CAU)](https://github.com/DeepthiSudharsan/htmltotext)")
st.subheader('Select the method of input:')
# Adding radio buttons for the user to choose between Uploading code or pasting the code
option = st.selectbox('Pick one', ['Click to view Dropdown Menu','Paste the HTML code', 'Upload a file'])
# Class definition
class html2Text:
    def __init__(self):
        self.state = 0
        self.s = ""
    def htm2Text(self, html):
        html = self.remove_style(html)
        self.s = ""
        for ch in html:
            if self.state == '<':
                if ch != '>':
                    continue
                self.state = 0
            else:
                if ch == '<':
                    self.state = '<'
                    continue
                self.s += ch
        return self.s.strip()
    def read1(self, filename):
        # with open(filename, 'r') as sentences:
        # f = filename.read()
        # for sentences in filename:
        #     f = sentences.read()
        fin_string = ""
        for line in filename:
            fin_string += line
        return fin_string.strip()
    def remove_style(self, text):
        if '<style' in text:
            startIindex = text.find('<style')
            end_index = text.find('</style>')
            #remove the style tag
            text = text.replace(text[startIindex:end_index+9], "")
            
        if '<script' in text:
            startIindex = text.find('<script')
            end_index = text.find('</script>')
            text = text.replace(text[startIindex:end_index+9], "")
            
        if '<!--' in text:
            startIindex = text.find('<!--')
            end_index = text.find('-->')            
            #remove the style tag
            text = text.replace(text[startIindex:end_index+9], "")        
        return(text)
# if user chooses to paste the data
if option == "Paste the HTML code":
    data = st.text_area('Paste HTML code in the text area given below (Once pasted/typed, click Ctrl + Enter to submit)')
    if data is not "":
        # flag = "True"
        ht = html2Text()
        with st.spinner(text = 'HTML to text conversion in progress'):
            opt = st.radio('Pick one', ['Raw Output', 'Cleaned Output (Without empty lines)'])
            if opt == "Raw Output":
                st.text_area(label = "Converted Text",value = ht.htm2Text(data), height = 350)
            else:
                st.text_area(label = "Converted Text",value = re.sub(r'\n\s*\n', '\n', ht.htm2Text(data)), height = 350)
            time.sleep(2)
            st.success('Converted')
    else:
        st.warning("Paste HTML code")

# if the user chooses to upload the data
elif option == "Upload a file":
    file = st.file_uploader('Upload file with HTML code')
    # browsing and uploading the dataset (strictly in csv format)
    if file is not None:
            ht = html2Text()
            stringio = StringIO(file.getvalue().decode("utf-8"))
            data = ht.read1(stringio)
            st.text_area(label = "Original HTML code",value = data, height = 350)
            with st.spinner(text = 'HTML to text conversion in progress'):
                opt = st.radio('Pick one', ['Raw Output', 'Cleaned Output (Without empty lines)'])
                if opt == "Raw Output":
                    st.text_area(label = "Converted Text",value = ht.htm2Text(data), height = 350)
                else:
                    st.text_area(label = "Converted Text",value = re.sub(r'\n\s*\n', '\n', ht.htm2Text(data)), height = 350)
                time.sleep(2)
                st.success('Converted')
    else:
        st.warning("Empty or No file uploaded")
