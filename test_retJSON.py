df2 = pd.DataFrame(out)

retJSON = {"selected_grades": json.loads(df2.to_json(orient="records"))}

print(retJSON)
