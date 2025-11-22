let
    // TODO: Update Server and Database names to match your environment
    ServerName = "sql-goeng-prod.database.windows.net",
    DatabaseName = "db-goeng-icp-prod",
    Source = Sql.Database(ServerName, DatabaseName, [Query="SELECT * FROM dbo.account_playbooks"])
in
    Source
