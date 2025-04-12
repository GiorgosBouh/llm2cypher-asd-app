# ... (all previous imports and configuration remain exactly the same until the file upload section)

    # === File Upload Section ===
    st.header("📄 Upload New Case")
    uploaded_file = st.file_uploader("Upload CSV for single child prediction", type="csv", key="file_uploader")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, delimiter=";")
            if not validate_csv(df):
                st.stop()

            if len(df) != 1:
                st.error("Please upload exactly one row (one child)")
                st.stop()

            st.subheader("👀 CSV Preview")
            st.dataframe(df.T)

            row = df.iloc[0]
            upload_id = str(uuid.uuid4())

            # Insert case
            with st.spinner("Inserting case into graph..."):
                insert_user_case(row, upload_id)

            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                run_node2vec()
                time.sleep(3)  # Give time for embeddings to generate

            # Check if case exists and get embedding
            with st.spinner("Verifying data..."):
                embedding = extract_user_embedding(upload_id)
                if embedding is None:
                    st.error("Failed to generate embedding")
                    st.stop()

            # ASD Prediction
            if 'asd_model' in st.session_state:
                with st.spinner("Predicting ASD traits..."):
                    model = st.session_state['asd_model']
                    proba = model.predict_proba([embedding])[0][1]

                    st.subheader("🛠️ Prediction Threshold")
                    threshold = st.slider(
                        "Select prediction threshold",
                        min_value=0.3,
                        max_value=0.9,
                        value=0.5,
                        step=0.01,
                        key="threshold_slider"
                    )

                    prediction = "YES (ASD Traits Detected)" if proba >= threshold else "NO (Control Case)"

                    st.subheader("🔍 Prediction Result")
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", prediction)
                    col2.metric(
                        "Confidence", 
                        f"{proba:.1%}" if prediction == "YES (ASD Traits Detected)" else f"{1 - proba:.1%}"
                    )

                    fig = px.bar(
                        x=["Control", "ASD Traits"],
                        y=[1 - proba, proba],
                        labels={'x': 'Class', 'y': 'Probability'},
                        title="Prediction Probabilities"
                    )
                    st.plotly_chart(fig)

            # Anomaly Detection - Fixed with proper handling
            with st.spinner("Checking for anomalies..."):
                try:
                    anomaly_result = train_isolation_forest()
                    
                    if anomaly_result and len(anomaly_result) == 2:  # Ensure we have both model and scaler
                        iso_forest, scaler = anomaly_result
                        embedding_scaled = scaler.transform([embedding])
                        anomaly_score = iso_forest.decision_function(embedding_scaled)[0]
                        is_anomaly = iso_forest.predict(embedding_scaled)[0] == -1

                        st.subheader("🕵️ Anomaly Detection")
                        if is_anomaly:
                            st.warning(f"⚠️ Anomaly detected (score: {anomaly_score:.3f})")
                            st.markdown("""
                            **Interpretation:**  
                            This case appears unusual compared to others in our database.  
                            Please review carefully as it may represent:
                            - A rare presentation of ASD traits
                            - Uncommon demographic combinations
                            - Potentially incomplete or unusual data
                            """)
                        else:
                            st.success(f"✅ Normal case (score: {anomaly_score:.3f})")
                            st.markdown("This case appears typical compared to others in our database.")

                        # Show distribution of anomaly scores
                        all_embeddings = get_existing_embeddings()
                        if all_embeddings is not None:
                            all_embeddings_scaled = scaler.transform(all_embeddings)
                            scores = iso_forest.decision_function(all_embeddings_scaled)

                            fig = px.histogram(
                                x=scores,
                                nbins=20,
                                labels={'x': 'Anomaly Score'},
                                title="Anomaly Score Distribution",
                                color_discrete_sequence=['#636EFA']
                            )
                            fig.add_vline(
                                x=anomaly_score, 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text="Current Case",
                                annotation_position="top"
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig)
                    else:
                        st.warning("""
                        Anomaly detection requires at least 10 existing cases in the database.
                        Currently we don't have enough data to perform this analysis.
                        """)
                except Exception as e:
                    st.error(f"Anomaly detection failed: {str(e)}")
                    logger.error(f"Anomaly detection error: {str(e)}")

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or corrupt")
            logger.error("Empty file uploaded")
        except pd.errors.ParserError:
            st.error("Could not parse the CSV file. Please check the format.")
            logger.error("CSV parsing error")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"File processing error: {str(e)}")

if __name__ == "__main__":
    main()