# Forecasting-App
Build consensus forecasts with sophisticated time-series algorithms using a simple and powerful Streamlit UI.

## Current Workflow

### Creating Input Dataframe

The user has access to all the warehouses, databases, schemas, and tables given in their Snowflake account through the UI. The user can select any given table and choose any subset of columns from that table. Furthermore, the user can choose to filter on values in these columns. The result is a dataframe.

### Feeding to Facebook Prophet

The user can create a Facebook Prophet model to be used for forecasting by selecting "Create Forecast."

### Outputs

The user can view the predictions as either a plotly chart or an embedded dataframe. Furthermore, they can download the dataframe.

## Future Plans

### Core Changes

The following changes will drastically change and improve the application. They will be implemented over the long run.

1. **Migrating Away from Prophet:** FBProphet is very easy to use but ultimately not a powerful time-series forecasting tool. I will be migrating the prediction workflow to either Darts or Tsfresh.

2. **Migrating Away from Streamlit:** Again, Streamlit is very powerful, but not great at handling user interaction. Every user interaction causes every single component to re-render. This not only makes it a hassle (or impossible) to save component state, but can also result in unnecessary and expensive queries without careful caching. We can use a powerful front end framework such as React.js instead.

### Smaller Changes

These are notes to myself on features to add that are much easier to implement and can be done in the short run.

1. Enable users to input their own SQL query to generate an input dataframe.

2. Create a multi-page view: Creating Input dataframe -> Specifying parameters for forecast -> Viewing/Ingesting outputs. 

3. Upgrade Python and other packages to be compatible with Streamlit's AgGrid package.
