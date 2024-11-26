import streamlit as st
import pandas as pd 
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #Facebook AI Similarity Search
from dotenv import load_dotenv
load_dotenv()
#define llm

import warnings
warnings.filterwarnings("ignore")
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


st.title('moodsandfoods')

st.header("Hello, there. ")

st.subheader("feeling blue?")
st.subheader("a little under the weather?")
st.subheader("moodsandfoods got you covered")
st.write("---")
# retrieve from vector database before anything else (beginning of app before user input)
# vectorization of cook book also saved in streamlit app not happening in real time

def retrieve_from_vector_db(vector_db_path):
    """
    this function splits out a retriever object from a local vector database
    """
    # instantiate embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2'
    )
    recipe_vectorstore = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = recipe_vectorstore.as_retriever()
    return retriever

recipe_retriever = retrieve_from_vector_db("../vector_databases/vector_db_recipe")
 
#-----


with open(file='vec.pkl',mode='br') as file:
    vec = pickle.load(file)

# take user input 
st.sidebar.subheader("Backlog")
date_previously = st.sidebar.date_input("Select and see recipes of previous days, moods and foods")
st.sidebar.write("---")
todays_date = st.sidebar.date_input("today")
st.sidebar.write("---")
user_mood_range = st.sidebar.slider('mood intensity', min_value=0, max_value=7)
color = st.sidebar.color_picker("mood color")


#user_mood_symptoms = st.sidebar.selectbox('Do you feel physically unwell?', options=['Headache', 'Weakness', 'Bloated'])
#user_mood_text = st.text_input("Mood")

# Mood input
user_mood_suggestion = st.multiselect("What's your mood today?", 
                                      ['boredom', 'unfocused', 'unmotivated', 'disinterest', 'depressed', 'confusion'])

user_mood_suggestion_str = ", ".join(user_mood_suggestion)

# Only show the next section if a mood is selected
if user_mood_suggestion:
    st.sidebar.subheader("moods")
    st.sidebar.button("boredom")
    st.sidebar.button("unfocused")
    st.sidebar.button("unmotivated")
    st.write("Thanks for indicating your mood. We are sorry that you feel that way! Let's see what we can do about that. Do you want to know which nutrients might be deficient or surplus and get a recipe based on that?")
    clicked = st.button("Tell me about those nutrients!")

    if clicked:
        st.write("Processing your input and providing suggestions!")


    if clicked:
        # transform user input with countvectorizer 
        # vectorize it
        user_mood_suggestion_vector = vec.transform([' '.join(user_mood_suggestion)])

        # call matrix_def
        matrix_def = pd.read_csv("full_matrix_def.csv", index_col=0)
        # call matrix_sur
        matrix_sur = pd.read_csv("full_matrix_sur.csv", index_col=0)
        matrix_sur = matrix_sur[~matrix_sur.index.str.lower().str.contains(r'water|vitamin d')]

        from sklearn.metrics.pairwise import cosine_similarity
        # Calculate cosine similarity between the user vector and each nutrient's mood profile
        #def
        similarity_def = cosine_similarity(matrix_def, user_mood_suggestion_vector)
        # Convert similarity scores to a DataFrame for readability
        similarity_def_df = pd.DataFrame(similarity_def, index=matrix_def.index, columns=["Similarity"])
        # Sort by similarity score to find the best matches
        similarity_def_df_sorted = similarity_def_df.sort_values(by="Similarity", ascending=False)
        # save top 3 results
        top_3_nutrients_def_df = similarity_def_df_sorted.head(3)
        top_3_nutrients_def_list = [str(item) for item in top_3_nutrients_def_df.index.tolist()]
        # Capitalize each nutrient and create a formatted list
        formatted_nutrients_def = ", ".join([nutrient.title() for nutrient in top_3_nutrients_def_list])

        # Display the output
        st.subheader("You might have a lack of one of these nutrients:")
        st.subheader(f"{formatted_nutrients_def}")

        #sur
        similarity_sur = cosine_similarity(matrix_sur, user_mood_suggestion_vector)
        # Convert similarity scores to a DataFrame for readability
        similarity_sur_df = pd.DataFrame(similarity_sur, index=matrix_sur.index, columns=["Similarity"])
        # Sort by similarity score to find the best matches
        similarity_sur_df_sorted = similarity_sur_df.sort_values(by="Similarity", ascending=False)
        # save top 3 results
        top_3_nutrients_sur_df = similarity_sur_df_sorted.head(3)
        top_3_nutrients_sur_list = [str(item) for item in top_3_nutrients_sur_df.index.tolist()]
        # Capitalize each nutrient and create a formatted list
        formatted_nutrients_sur = ", ".join([nutrient.title() for nutrient in top_3_nutrients_sur_list])
        st.subheader("You might have a surplus of one of these nutrients:")
        st.subheader(f"{formatted_nutrients_sur}")


        #st.write("Based on this, we suggest the following recipe:")


        # get food input 2 lists of ingredients 
        # call food ingredient input
        food_dict_df = pd.read_csv("food_dict_refined.csv")
        # Assuming `df` is your DataFrame
        food_dict = food_dict_df.to_dict()

        # list of list to list
        food_contents_to_add = list(set([item for k, v in food_dict.items() if k in top_3_nutrients_def_df.index for item in v]))
        food_contents_to_add = [str(item) for item in food_contents_to_add]
        food_contents_not_to_add = list(set([item for k, v in food_dict.items() if k in top_3_nutrients_sur_df.index for item in v]))
        food_contents_not_to_add = [str(item) for item in food_contents_not_to_add]

        # get cook book from database

        # vectorize and save vector beforehand to apply here 

        # chain passing user inquiry to retriever object 
        from langchain import hub
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains.retrieval import create_retrieval_chain
        def connect_chains(retriever):
            """
            this function connects stuff_documents_chain with retrieval_chain
            """
            stuff_documents_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=hub.pull("langchain-ai/retrieval-qa-chat")
            )
            retrieval_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=stuff_documents_chain
            )
            return retrieval_chain

        recipe_retrieval_chain = connect_chains(recipe_retriever)

        # convert it to a nice text output 
        #output = recipe_retrieval_chain.invoke(
            #{f"input": "Give me the name and detailed description of the recipe suggestion that contains foods like {food_contents_to_add} and does not contain {food_contents_not_to_add}."}
        #)
        #st.markdown("### **Suggested Recipe**")
        #st.markdown(f"**Recipe:** {output['answer']}")

    click_for_recipe = st.button("What can I cook to improve my mood?")

    if click_for_recipe:
        #---
        user_mood_suggestion_vector = vec.transform([' '.join(user_mood_suggestion)])

        # call matrix_def
        matrix_def = pd.read_csv("full_matrix_def.csv", index_col=0)
        # call matrix_sur
        matrix_sur = pd.read_csv("full_matrix_sur.csv", index_col=0)
        # correct the CSV
        matrix_sur = matrix_sur[~matrix_sur.index.str.lower().str.contains(r'water|vitamin d')]


        from sklearn.metrics.pairwise import cosine_similarity
        # Calculate cosine similarity between the user vector and each nutrient's mood profile
        #def
        similarity_def = cosine_similarity(matrix_def, user_mood_suggestion_vector)
        # Convert similarity scores to a DataFrame for readability
        similarity_def_df = pd.DataFrame(similarity_def, index=matrix_def.index, columns=["Similarity"])
        # Sort by similarity score to find the best matches
        similarity_def_df_sorted = similarity_def_df.sort_values(by="Similarity", ascending=False)
        # save top 3 results
        top_3_nutrients_def_df = similarity_def_df_sorted.head(3)
        top_3_nutrients_def_list = [str(item) for item in top_3_nutrients_def_df.index.tolist()]
        # Capitalize each nutrient and create a formatted list
        formatted_nutrients_def = ", ".join([nutrient.title() for nutrient in top_3_nutrients_def_list])

        # Display the output
        st.subheader("You might have a lack of one of these nutrients:")
        st.subheader(f"{formatted_nutrients_def}")

        #sur
        similarity_sur = cosine_similarity(matrix_sur, user_mood_suggestion_vector)
        # Convert similarity scores to a DataFrame for readability
        similarity_sur_df = pd.DataFrame(similarity_sur, index=matrix_sur.index, columns=["Similarity"])
        # Sort by similarity score to find the best matches
        similarity_sur_df_sorted = similarity_sur_df.sort_values(by="Similarity", ascending=False)
        # save top 3 results
        top_3_nutrients_sur_df = similarity_sur_df_sorted.head(3)
        top_3_nutrients_sur_list = [str(item) for item in top_3_nutrients_sur_df.index.tolist()]
        # Capitalize each nutrient and create a formatted list
        formatted_nutrients_sur = ", ".join([nutrient.title() for nutrient in top_3_nutrients_sur_list])
        st.subheader("You might have a surplus of one of these nutrients:")
        st.subheader(f"{formatted_nutrients_sur}")


        #st.write("Based on this, we suggest the following recipe:")


        # get food input 2 lists of ingredients 
        # call food ingredient input
        food_dict_df = pd.read_csv("food_dict_refined.csv")
        # Assuming `df` is your DataFrame
        food_dict = food_dict_df.to_dict()

        # list of list to list
        food_contents_to_add = list(set([item for k, v in food_dict.items() if k in top_3_nutrients_def_df.index for item in v]))
        food_contents_to_add = [str(item) for item in food_contents_to_add]
        food_contents_not_to_add = list(set([item for k, v in food_dict.items() if k in top_3_nutrients_sur_df.index for item in v]))
        food_contents_not_to_add = [str(item) for item in food_contents_not_to_add]

        # get cook book from database

        # vectorize and save vector beforehand to apply here 

        # chain passing user inquiry to retriever object 
        from langchain import hub
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains.retrieval import create_retrieval_chain
        def connect_chains(retriever):
            """
            this function connects stuff_documents_chain with retrieval_chain
            """
            stuff_documents_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=hub.pull("langchain-ai/retrieval-qa-chat")
            )
            retrieval_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=stuff_documents_chain
            )
            return retrieval_chain

        recipe_retrieval_chain = connect_chains(recipe_retriever)

        # convert it to a nice text output 
        #output = recipe_retrieval_chain.invoke(
            #{f"input": "Give me the name and detailed description of the recipe suggestion that contains foods like {food_contents_to_add} and does not contain {food_contents_not_to_add}."}
        #)
        #st.markdown("### **Suggested Recipe**")
        #st.markdown(f"**Recipe:** {output['answer']}")
        #---
        ## new script for presentation 
        # Define prompt template 
        from langchain.prompts.prompt import PromptTemplate

        query = f"""
            Say that feeling {user_mood_suggestion_str} is really annoying and that you are here to help with a flavorful fun recipe. 
            1. Find a suitable fun recipe that does not contain meat but contains foods that are rich in {formatted_nutrients_def} and do not contain {formatted_nutrients_sur}.
            2. Give the title and recipe including all steps (ingredients, how to cook) and highlight the importance of {formatted_nutrients_def} rich ingredients.
            3. Mention that the recipe does not contain {formatted_nutrients_sur} for your own good.
            4. Give top 3 interesting facts on why it is important for you to increase {formatted_nutrients_def} intake and why to avoid {formatted_nutrients_sur} to fight feeling {user_mood_suggestion_str} and balance the mood.
            5. Say that you hope that the recipe is helpful to feel better soon. 
            Make sure that paragraph 4.is clearly separated and has a title.
            """
        prompt_template = PromptTemplate(
        input_variables=[f"{formatted_nutrients_def}, {formatted_nutrients_sur}, {user_mood_suggestion_str}"],
        template=query
        )

            #Define chain

            # allows to link the output of one LLM call as the input of another
            # The `|` symbol chains together the different components, feeding the output from one component as input into the next component.
            # In this chain the user input is passed to the prompt template, then the prompt template output is passed to the model. 
        chain = prompt_template | llm

            # invoke chain 

        text_data ="""
        The recipe contains relevant nutrients that help balancing your mood. It contains exactly whats needed in your specific case. 
        """
        output = chain.invoke(input={"food_contents_to_add": text_data})
        
        st.markdown(f"**Recipe:** {output.content}")
            # make answer beautiful ? 

        st.subheader("Do you like this recipe?")
        
        st.feedback("stars")

        st.text_area("Any feedback is highly appreciated!")


        clicked_again = st.button("If you don't like the recipe, let's generate another one... ")

        st.write("---")
        
        st.header("moodsandfoods news")
        st.button("Subscribe to our newsletter!")
        st.write("---")
        st.subheader("Launching with grocery delivery soon....")
        st.write("Ever craved a recipe of moodsandfoods that much that you didn't even have the time to run to the supermarket close by? Also here, soon, moodsandfoods has got you covered. \n moodsandfoods is launching their cooperation with different grocery delivery providers in Berlin. \n Stay supercharged and in a good mood... ")
        st.write("---")
        st.subheader("Integrating into femtech cycle tracker apps....")
        st.write("Founder and CEO of mood based recipe generator recently had first talks with femtech platforms to provide moodsandfoods' algorithms personnalise them even more especially for women. \n Ever wondered if your mood and eating behavior might be interrelated with your cycle? Well moodsandfoods thinks so too...")
        st.write("---")


