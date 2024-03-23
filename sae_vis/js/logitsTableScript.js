// ! Using `DATA.logitsTableData` to fill in the pos/neg logit tables

function setupLogitTables(logitsTableId, tablesData, tableMetaData) {

    // Fetch table data (singular)
    const tableData = tablesData[tableMetaData.dataKey];

    // Select the table container by its ID
    const tablesContainer = d3.select(`#${logitsTableId}`);

    // Select the object that contains this particular table (negative or positive)
    const tableId = `${tableMetaData.class}-${logitsTableId}`;
    const tableContainer = tablesContainer.select(`#${tableId}`);
    
    // If this table container doesn't exist, create it
    if (tableContainer.empty()) {
        const section = tablesContainer.append("div").attr("class", tableMetaData.class);
        section.append("h4").html(tableMetaData.title);
        section.append("table").attr("id", tableId).attr("class", "table-left");
    }

    // Now we can select the table by its ID, empty data, and add new data
    const table = tablesContainer.select(`#${tableId}`);
    table.selectAll('tr').remove();

    // Bind logits data to rows
    const rows = table.selectAll('tr')
        .data(tableData)
        .enter()
        .append('tr');
    
    // Append token cell
    rows.append('td')
        .attr('class', 'left-aligned')
        .append('code')
        .style('background-color', d => d.color)
        .text(d => d.symbol);
    
    // Append value cell
    rows.append('td')
        .attr('class', 'right-aligned')
        .text(d => d.value.toFixed(2));
}

// Define some data which will tell us how to create our tables (this isn't a function of data, it's always the same)
var logitTablesMetaData = [
    {title: "NEGATIVE LOGITS", dataKey: "negLogits", class: "negative"},
    {title: "POSITIVE LOGITS", dataKey: "posLogits", class: "positive"},
]

// 'logitsTableData' is a dictionary mapping suffixes to logits table data (to make each logits table unique)
// We iterate over it, and create a logits table for each one
Object.entries(DATA.logitsTableData).forEach(([suffix, tablesData]) => {
    var t0 = performance.now();

    // For each set of (pos / neg) table pairs, we need to empty each of them, then create them again
    // tablesContainerId = `feature-tables-${suffix}`;
    // setupLogitTables(logitsTableId, "negative", tableData.negLogits);
    // setupLogitTables(logitsTableId, "positive", tableData.posLogits);
    
    logitsTableId = `logits-table-${suffix}`;
    logitTablesMetaData.forEach(tableMetaData => {
        setupLogitTables(logitsTableId, tablesData, tableMetaData);
    });

    var t1 = performance.now();
    console.log(`HTML for ${logitsTableId} generated in ${(t1 - t0).toFixed(1)} ms`);
});




