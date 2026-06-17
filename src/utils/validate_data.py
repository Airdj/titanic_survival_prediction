import great_expectations as ge

def validate_loan_data(df, columns):
    print('Starting data validation with Great Expectations')
    print(ge.__version__)
    context = ge.get_context()

    data_source_name = 'my_data_source'
    data_source = context.data_sources.add_pandas(name=data_source_name)
    data_asset_name = 'my_data_asset_name'
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition_name = 'my_batch_definition'
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name=batch_definition_name)
    batch_parameters = {'dataframe': df}

    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    suite = context.suites.add(ge.ExpectationSuite(name='titanic_expectation_suite'))
    #schema validation - essential columns
    for col in columns:
        exceptation1 = ge.expectations.ExpectColumnToExist(column=col)
        suite.add_expectation(exceptation1)
    # bussines logic validation
    my_expectation_list = [

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Pclass',
            value_set=[1,2,3]
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Sex',
            value_set=['male', 'female']
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="Age",
            min_value=0.0,
            max_value=100.0
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='SibSp',
            value_set=['0','1','2','3','4','5','6','7','8']
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Parch',
            value_set=['0','1','2','3','4','5','6']
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="Fare",
            min_value=0,
            max_value=1000
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Embarked',
            value_set=['S', 'C','Q']
        ),
       ]

    for expectation in my_expectation_list:
        suite.add_expectation(expectation)

    # run validation
    print('running validation')
    validation_definition = ge.ValidationDefinition(data=batch_definition, suite=suite, name='validation_batch_definition')
    validation_definition = context.validation_definitions.add(validation_definition)
    results = validation_definition.run(batch_parameters=batch_parameters)

    #process results
    failed_expectations = []
    failed_columns = []
    for r in results['results']:
        if not r['success']:
            expectation_type = r['expectation_config']['type']
            expectation_column = r['expectation_config']['kwargs']['column']
            failed_expectations.append(expectation_type)
            failed_columns.append(expectation_column)
    #print validation summary
    total_checks = len(results['results'])
    passed_checks = sum(1 for r in results['results'] if r['success'])
    failed_checks = total_checks - passed_checks

    if results['success']:
        print(f'Data validation PASSED: {passed_checks}/{total_checks} checks successful')
    else:
        print(f'Data validation FAILED: {failed_checks}/{total_checks} checks failed')
        print(f'Failed expectations: {failed_expectations}')
        print(f'Failed columns: {failed_columns}')
    return results['success'], failed_expectations