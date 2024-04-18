// ! Using `DATA.featureTablesData` to create & fill in the feature tables

function setupFeatureTables(tablesContainerId, tablesData, tableMetaData) {
    
    // Fetch table data (singular), if it doesn't exist then return early
    const tableData = tablesData[tableMetaData.dataKey] || null;
    if (tableData === null) {
        return;
    }

    // Get table container from its ID
    const featureTablesDiv = d3.select(`#${tablesContainerId}`);

    // Add a div to hold the header and table, give it a unique ID, and empty it (if that div already existed)
    const tableContainerId = `${tableMetaData.dataKey}-${tablesContainerId}`;
    const tableContainer = featureTablesDiv.append("div").attr("id", tableContainerId);
    tableContainer.selectAll("*").remove();

    // Add the title
    tableContainer.append("h4").html(tableMetaData.title);

    // Create the table, and header / body
    const table = tableContainer.append("table").attr("class", "table-left");
    const thead = table.append("thead");
    const tbody = table.append("tbody");
    const headerRow = thead.append("tr");

    // Append the 3 columns to this table's header row
    tableMetaData.columns.forEach(col => {
        headerRow.append("td")
        .attr("class", col === "Index" ? "left-aligned" : "right-aligned")
        .html(col);
    });

    // Append our data to the tbody
    const rows = tbody.selectAll('tr')
        .data(tableData)
        .enter()
        .append('tr');

    rows.selectAll('td')
        .data(row => Object.values(row))
        .enter()
        .append('td')
        .attr('class', (d, i) => i === 0 ? 'left-aligned' : 'right-aligned')
        .html(d => d);
}
    
// Define some data which will tell us how to create our tables (this isn't a function of data, it's always the same)
const featureTablesMetaData = [
    {title: "NEURON ALIGNMENT", columns: ["Index", "Value", "% of L<sub>1</sub>"], dataKey: "neuronAlignment"},
    {title: "CORRELATED NEURONS", columns: ["Index", "Pearson Corr.", "Cosine Sim."], dataKey: "correlatedNeurons"},
    {title: "CORRELATED FEATURES", columns: ["Index", "Pearson Corr.", "Cosine Sim."], dataKey: "correlatedFeatures"},
    {title: "CORRELATED FEATURES (B-ENCODER)", columns: ["Index", "Pearson Corr.", "Cosine Sim."], dataKey: "correlatedFeaturesB"},
]

// 'featureTablesData' is a dictionary mapping suffixes to feature tables data (to make each table unique)
// We iterate over it, and create a table for each one
Object.entries(DATA.featureTablesData).forEach(([suffix, tablesData]) => {
    var t0 = performance.now();

    // For each set of tables, we need to empty it, then create each of the new tables (up to 3)
    tablesContainerId = `feature-tables-${suffix}`;
    featureTablesMetaData.forEach(tableMetaData => {
        setupFeatureTables(tablesContainerId, tablesData, tableMetaData);
    });
    
    var t1 = performance.now();
    console.log(`HTML for ${tablesContainerId} generated in ${(t1 - t0).toFixed(1)} ms`);
});



