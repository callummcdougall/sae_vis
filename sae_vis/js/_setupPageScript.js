// The start key has already been user-defined, we need to check it's present in DASHBOARD_DATA
if (!Object.keys(DASHBOARD_DATA).includes(START_KEY)) {
    console.error(`No data available for key: ${START_KEY}`);
}
const startParts = START_KEY.split('|');

// Function which will measure the width of a key (used for dynamically setting the width of our dropdowns)
function measureTextWidth(text, font) {
    // Create a temporary span element
    const span = document.createElement("span");
    span.style.visibility = "hidden"; // Make sure it's not visible
    span.style.position = "absolute"; // Take it out of document flow
    span.style.font = font; // Apply font styling similar to the select options
    span.textContent = text;
    document.body.appendChild(span);
    const width = span.offsetWidth; // Get the width of the span
    document.body.removeChild(span); // Remove the span from the body
    return width;
}

// Check if there's only one key, in which case we have no selection to make
if (Object.keys(DASHBOARD_DATA).length === 1) {

    const onlyKey = Object.keys(DASHBOARD_DATA)[0];
    createVis(DASHBOARD_DATA[onlyKey]);

} else {

    // Assuming `DASHBOARD_DATA` is available and d3 has been included
    const parsedKeys = Object.keys(DASHBOARD_DATA).map(key => key.split("|"));

    // Determine `n` - the number of dropdowns needed
    const n = parsedKeys[0].length;

    // Create a structure to store options for each dropdown
    const options = Array.from({ length: n }, () => new Set());

    // Populate options for each dropdown
    parsedKeys.forEach(parts => {
        parts.forEach((part, index) => options[index].add(part));
    });

    // Select the container for the dropdowns
    const container = d3.select('#dropdown-container');

    // Create the dropdowns
    options.forEach((opts, i) => {
        // We wrap each `select` element in a `.select` div, for styling reasons)
        const selectDiv = container.append('div').attr('class', 'select');
        const selectElem = selectDiv.append('select').attr('id', `select-${i}`);
        let maxWidth = 0;

        // Set the title of this dropdown
        

        opts.forEach(opt => {
            // Add this as an option
            const optionElem = selectElem.append('option').text(opt).attr('value', opt);

            // If it matches the start key, set it to true
            if (startParts[i] === opt) {
                optionElem.attr('selected', true);
            }

            // Calculate the width of this option, and possibly update the max width (for the select div)
            const width = measureTextWidth(opt, "1em system-ui");
            if (width > maxWidth) {
                maxWidth = width;
            }
        });

        // Set the width of the select div to the max width + 45px (for the dropdown arrow)
        selectDiv.style('width', `${maxWidth + 45}px`);

        // Add event listener to update the visualization (and the selection options) when the dropdown changes
        selectElem.on('change', function() {
            updateDropdowns();
        });
    });

    // This gets called every time the dropdowns are updated
    function updateDropdowns() {

        // Empty all grid-column elements
        d3.selectAll(".grid-column").each(function() {
            d3.select(this).selectAll("*").html(""); 
        });

        // Get the current value of each dropdown
        const currentSelections = options.map((_, i) => d3.select(`#select-${i}`).property('value'));

        // Disable options that are not available based on the current selections
        options.forEach((opts, i) => {
            const select = d3.select(`#select-${i}`);
            select.selectAll('option').each(function() {
                const optionValue = d3.select(this).attr('value');
                const potentialKey = [...currentSelections.slice(0, i), optionValue, ...currentSelections.slice(i + 1)].join('|');
                d3.select(this).attr('disabled', !DASHBOARD_DATA.hasOwnProperty(potentialKey) ? true : null);
            });
        });

        // Parse these options into the key for our DASHBOARD_DATA object
        const selectedKey = currentSelections.join('|');

        // Create vis using that data
        createVis(DASHBOARD_DATA[selectedKey]);
    }

    // Initial trigger
    updateDropdowns();
}