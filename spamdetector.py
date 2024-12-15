import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl', 'rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham")

    st.subheader("Classification")
    user_input = st.text_area("Enter an email to classify")

    if st.button("Classify"):
        if user_input:
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == 0:
                st.success("This is Not a Spam Email")
            else:
                st.error("This is a Spam Email")
        else:
            st.write("Please enter an email to classify.")

if __name__ == '__main__':
    main()

