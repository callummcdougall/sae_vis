// ! Using `DATA.gridColumTitlesData` to fill in the titles for each column

if (Object.keys(DATA).includes("gridColumnTitlesData")) {

    // Iterate through the titles we've been given
    Object.entries(DATA.gridColumnTitlesData).forEach(([columnIdx, title]) => {

        // Select the correct title-containing div
        var titleDiv = d3.select(`#column-${columnIdx}-title`);

        // Empty this div, and fill it with new title
        titleDiv.selectAll('*').remove();
        titleDiv.html(title);
    });
}