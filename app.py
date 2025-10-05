if st.checkbox("Show Confusion Matrix for Best Model"):
        best_model = models[best_model_name]
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

st.subheader("Optional Neural Network Training")
if st.checkbox("Train Neural Network"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train_scaled, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
    test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    st.success(f"Neural Network Test Accuracy: {test_acc:.2f}")
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[0].legend()
    ax[0].set_title("Accuracy Over Epochs")
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].legend()
    ax[1].set_title("Loss Over Epochs")
    st.pyplot(fig)

st.header("Optional KMeans + PCA Clustering")
if st.checkbox("Run KMeans Clustering"):
    num_clusters = st.slider("Select number of clusters", 2, 10, 3)
    scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled)
    df['Cluster'] = labels
    pca = PCA(2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(data=components, columns=["PC1","PC2"])
    pca_df['Cluster'] = labels
    fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
                     title="KMeans Clustering with PCA")
    st.plotly_chart(fig, use_container_width=True)

st.header("Sentiment Analysis")
text_input = st.text_area("Enter text to analyze sentiment:")
if st.button("Analyze Sentiment"):
    if text_input.strip():
        sentiment = TextBlob(text_input).sentiment.polarity
        if sentiment > 0:
            st.success(f"Positive Sentiment ({sentiment:.2f})")
        elif sentiment < 0:
            st.error(f"Negative Sentiment ({sentiment:.2f})")
        else:
            st.warning("Neutral Sentiment (0.00)")
    else:
        st.warning("Please enter valid text.")

st.markdown("---")
st.markdown("Developed for AI Mental Health Research Dashboard")





