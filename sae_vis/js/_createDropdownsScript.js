
// The start key has already been user-defined, we need to check it's present in DATA
if (!Object.keys(DATA).includes(START_KEY)) {
    console.error(`No data available for key: ${START_KEY}`);
}
const startParts = START_KEY.split('|');

// Check if there's only one key, in which case we have no selection to make
if (Object.keys(DATA).length === 1) {

    const onlyKey = Object.keys(DATA)[0];
    createVis(DATA[onlyKey]);

} else {

    // Assuming `DATA` is available and d3 has been included
    const parsedKeys = Object.keys(DATA).map(key => key.split("|"));

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
        const select = container.append('select').attr('id', `select-${i}`);

        opts.forEach(opt => {
            // Add this as an option
            const optionElem = select.append('option').text(opt).attr('value', opt);

            // If it matches the start key, set it to true
            if (startParts[i] === opt) {
                optionElem.attr('selected', true);
            }
        });

        // Add event listener to update the visualization (and the selection options) when the dropdown changes
        select.on('change', function() {
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
                d3.select(this).attr('disabled', !DATA.hasOwnProperty(potentialKey) ? true : null);
            });
        });

        // Parse these options into the key for our DATA object
        const selectedKey = currentSelections.join('|');

        // Create vis using that data
        createVis(DATA[selectedKey]);
    }

    // Initial trigger
    updateDropdowns();
}