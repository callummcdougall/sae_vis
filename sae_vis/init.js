// ! [1/5] featureTables

function setupFeatureTables(featureIdx, componentData, containerId) {
    // Metadata for each feature table we'll set up
    const featureTablesMetaData = [
        {title: "NEURON ALIGNMENT", columns: ["Index", "Value", "% of L<sub>1</sub>"], dataKey: "neuronAlignment"},
        {title: "CORRELATED NEURONS", columns: ["Index", "Pearson Corr.", "Cosine Sim."], dataKey: "correlatedNeurons"},
        {title: "CORRELATED FEATURES", columns: ["Index", "Pearson Corr.", "Cosine Sim."], dataKey: "correlatedFeatures"},
        {title: "CORRELATED FEATURES (B-ENCODER)", columns: ["Index", "Pearson Corr.", "Cosine Sim."], dataKey: "correlatedFeaturesB"},
    ]
    featureTablesMetaData.forEach(tableMetaData => {
        // Fetch table data (singular), if it doesn't exist then return early
        const tableData = componentData[tableMetaData.dataKey] || null;
        if (tableData === null) {
            return;
        }

        // Add a div to hold the header and table, give it a unique ID, and empty it (if that div already existed)
        const tableContainerId = `${tableMetaData.dataKey}-featureTables`;
        const tableContainer = d3.select(`#${containerId}`).append("div").attr("id", tableContainerId);
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
    });
}

// ! [2/5] actsHistogram

function setupActsHistogram(featureIdx, componentData, containerId) {

    const histContainer = d3.select(`#${containerId}`).attr("class", "plotly-hist");

    // Create layout. This has 2 lines with annotations, both initially set to invisible (first is for dynamic on-hover
    // and second is for static, for the prompt-centric vis). We need to define them both so that when JavaScript alters
    // 'shapes[0]' or 'shapes[1]' we get no error
    var layout = {
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: {
            gridcolor: '#eee',
            zerolinecolor: '#eee',
            tickvals: componentData.ticks,
            range: [0, 1.2 * Math.max(...componentData.x)],
        },
        yaxis: {gridcolor: '#eee', zerolinecolor: '#eee'},
        barmode: 'relative',
        bargap: 0.01,
        showlegend: false,
        margin: {l: 50, r: 25, b: 25, t: 25, pad: 4},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        responsive: true,
        shapes: [createHistLine('black', 2, 1.0), createHistLine('black', 1, 0.9)],
        annotations: [createHistAnnotation(), createHistAnnotation()],
    };

    // Create traces (we also compute bar colors from x tickvalues here)
    var traces = [{
        x: componentData.x,
        y: componentData.y,
        type: 'bar',
        marker: {color: componentData.x.map(v => actColor((0.4 + 0.6 * v/Math.max(...componentData.x)), 1))},
    }];

    // Plot the histogram
    Plotly.newPlot(containerId, traces, layout, {responsive: true, displayModeBar: false});

    // Maybe add title
    if (componentData.title) {
        histContainer.node().parentNode.insertBefore(
            document.createElement('h4'), document.getElementById(containerId)
        ).innerHTML = componentData.title;
    }

}

// ! [3/5] logitsHistogram

function setupLogitsHistogram(featureIdx, componentData, containerId) {

    const histContainer = d3.select(`#${containerId}`).attr("class", "plotly-hist");

    // Create layout. This has 2 lines with annotations, both initially set to invisible (first is
    // for dynamic on-hover and second is for static, for the prompt-centric vis). We need to define 
    // them both so that when JavaScript alters 'shapes[0]' or 'shapes[1]' we get no error
    var layout = {
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: {
            gridcolor: '#eee',
            zerolinecolor: '#eee',
            tickvals: componentData.ticks,
            range: [1.2 * Math.min(...componentData.x), 1.2 * Math.max(...componentData.x)],
        },
        yaxis: {gridcolor: '#eee', zerolinecolor: '#eee'},
        barmode: 'relative',
        bargap: 0.01,
        showlegend: false,
        margin: {l: 50, r: 25, b: 25, t: 25, pad: 4},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        responsive: true,
        shapes: [createHistLine('black', 2, 1.0), createHistLine('black', 1, 0.9)],
        annotations: [createHistAnnotation(), createHistAnnotation()],
    };

    // Create traces. This is a bit messier than for acts_histogram_script.js, because we have 2 different colors
    traces = [
        {
            x: componentData.x.filter(value => value >= 0),
            y: componentData.y.filter((_, i) => componentData.x[i] >= 0),
            type: 'bar',
            marker: {color: 'rgba(0,0,255,0.5)'}
        },
        {
            x: componentData.x.filter(value => value < 0),
            y: componentData.y.filter((_, i) => componentData.x[i] < 0),
            type: 'bar',
            marker: {color: 'rgba(255,0,0,0.5)'}
        }
    ];

    // Plot the histogram
    Plotly.newPlot(containerId, traces, layout, {responsive: true, displayModeBar: false});

    // Maybe add title
    if (componentData.title) {
        histContainer.node().parentNode.insertBefore(
            document.createElement('h4'), document.getElementById(containerId)
        ).innerHTML = componentData.title;
    }
}

// ! [4/5] logitsTable & probeLogitsTables

function _setupLogitTables(tablesData, containerId, negTitle, posTitle) {
    // Define some data which will tell us how to create our tables (this isn't a function of data, it's always the same)
    const logitTablesMetaData = [
        {title: negTitle, dataKey: "negLogits", class: "negative"},
        {title: posTitle, dataKey: "posLogits", class: "positive"},
    ]

    // Select the table container by its ID
    const tablesContainer = d3.select(`#${containerId}`);

    // For each set of (pos / neg) table pairs, we need to empty each of them, then create them again
    logitTablesMetaData.forEach(tableMetaData => {
        
        // Fetch table data (singular)
        const tableData = tablesData[tableMetaData.dataKey];
    
        // Select the object that contains this particular table (negative or positive)
        const tableId = `${tableMetaData.class}-${containerId}`;
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
            .append('span')
            .style('background-color', d => lossColor(-d.value, tablesData.maxLogits))
            .text(d => d.symbol);
        
        // Append value cell
        rows.append('td')
            .attr('class', 'right-aligned')
            .text(d => d.value.toFixed(2));
    });
}

function setupLogitTables(featureIdx, componentData, containerId) {
    // Single logit table: neg & pos
    _setupLogitTables(componentData, containerId, "NEGATIVE LOGITS", "POSITIVE LOGITS");
}

function setupProbeLogitsTables(featureIdx, componentData, containerId) {
    const tablesContainer = d3.select(`#${containerId}`);

    // Logit table for every probe
    Object.entries(componentData).forEach(([probeName, tablesData]) => {
        tablesContainer.append("div")
            .attr("id", `${containerId}-${stringToId(probeName)}`)
            .attr("class", "logitsTable");
        
        _setupLogitTables(tablesData, `${containerId}-${stringToId(probeName)}`, probeName, "&nbsp;");
    })
}

// ! [5/5] sequences

function _setupSeqMultiGroupStandard(featureIdx, componentData, containerId) {

    const seqGroupsContainer = d3.select(`#${containerId}`)
    
    componentData.forEach(seqGroup => {
        const {seqGroupData, seqGroupMetadata} = seqGroup;
        const {seqGroupID, title, maxAct, maxLoss, maxDFA} = seqGroupMetadata;

        if (title) {
            seqGroupsContainer.append('div').html(`<h4>${title}</h4>`);
        }

        var seqGroupContainerInner;
        const seqGroupContainerOuter = seqGroupsContainer.append('div')
            .attr('id', seqGroupID)
            .attr('class', 'seq-group');
        
        if (seqGroupData[0].dfaSeqData.length > 0) {
            // If we're using DFA, then insert the DFA column on the left half, and define a thing on the right half to
            // insert the normal sequences in
            seqGroupContainerOuter.style('display', 'flex');

            const dfaSeqGroupContainer = seqGroupContainerOuter.append('div')
                .attr('class', 'dfa-seq-group')
                .style('flex', '1 1 50%');

            var seqGroupContainerInner = seqGroupContainerOuter.append('div')
                .attr('class', 'seq-group')
                .style('flex', '1 1 50%');

            dfaSeqGroupContainer.selectAll('.seq')
                .data(seqGroupData)
                .enter()
                .append('div')
                .attr('class', 'seq')
                .each(function({seqData, dfaSeqData}) {
                    dfaSeqData.forEach(tokData => {
                        generateTokHtmlElement({
                            parent: d3.select(this), 
                            tok: tokData.tok,
                            tokID: tokData.tokID,
                            tokPosn: tokData.tokPosn,
                            tokValue: tokData["dfaValue"],
                            isDfa: true,
                            isBold: tokData["isBold"],
                            maxValue: maxDFA,
                        });
                    });
                });
        } else {
            // If not using DFA, then we just insert the normal sequences into the container as given
            seqGroupContainerInner = seqGroupContainerOuter;
        }

        seqGroupContainerInner.selectAll('.seq')
                .data(seqGroupData)
                .enter()
                .append('div')
                .attr('class', 'seq')
                .each(function({seqData}) {
                    seqData.forEach(tokData => {
                        generateTokHtmlElement({
                            parent: d3.select(this),
                            maxLoss: maxLoss,
                            maxValue: maxAct,
                            tok: tokData.tok,
                            tokID: tokData.tokID,
                            tokPosn: tokData.tokPosn,
                            isBold: tokData["isBold"],
                            tokValue: tokData["featAct"],
                            tokenLogit: tokData["tokenLogit"],
                            lossEffect: tokData["lossEffect"],
                            logitEffect: tokData["logitEffect"],
                            origProb: tokData["origProb"],
                            newProb: tokData["newProb"],
                            posToks: tokData["posToks"],
                            posVal: tokData["posVal"],
                            negToks: tokData["negToks"],
                            negVal: tokData["negVal"],
                            permanentLine: tokData["permanentLine"],
                            hide: tokData["hide"]
                        });
                    });
                });
    });

    function generateTokHtmlElement({
        parent,
        maxLoss = 1.0,                     // for setting underline color
        maxValue = 1.0,                    // for setting background color
        tok,
        tokID,
        tokPosn,                           // position in dataset, as a string of ("{batch_idx}, {seq_idx}")
        tokValue = 0.0,                    // either feature activation or attribution value
        isDfa = false,                     // controls highlight color (dfa src is orange, normal is green)
        isBold = false,                    // is this token bolded?
        tokenLogit = 0.0,                  // raw logit effect on this token
        lossEffect = 0.0,                  // effect on loss
        logitEffect = 0.0,                 // effect on logits
        origProb = null,                   // orig probability (for hoverinfo)
        newProb = null,                    // new probability (for hoverinfo)
        posToks = [],                      // most-positive tokens (as strings)
        posVal = [],                       // most-positive token values
        negToks = [],                      // most-negative tokens (as strings)
        negVal = [],                       // most-negative token values
        permanentLine = false,             // do we show a permanent line on histogram?
        hide = false                       // do we suppress hoverdata for this token?
    }) {
        // Figure out if previous token was active (this affects the construction of the tooltip)
        let prevTokenActive = posToks.length + negToks.length > 0;
    
        // Create the token span (this will contain just the token, not the hoverdata)
        let tokenSpan = parent.append("span")
            .attr("class", "hover-text");
    
        // Put actual token in the tokenSpan object (i.e. the thing we see without hovering)    
        tokenSpan.append("span")
            .attr("class", "token")
            .style("background-color", actColor(tokValue, maxValue, isDfa))
            .style("border-bottom", `4px solid ${lossColor(lossEffect, maxLoss, opacity=1)}`)
            .style("font-weight", isBold ? "bold" : "normal")
            .text(tok);
    
        // If hide is true, then we just show a box saying "no information was calculated for this token"
        if (hide) {
            // First define the tooltip div (added to the parent element, i.e. it's a sibling of the token span)
            let tooltipHeight = 70;
            let tooltipWidth = 150;
            let tooltipDiv = parent.append("div")
                .attr("class", "tooltip")
                .style("height", tooltipHeight + "px")
                .style("width", tooltipWidth + "px")
                .style("font-size", "0.8em")
                .style("white-space", "pre-wrap")
                .style("align-items", "center")
                .style("text-align", "center")
                .style("padding", "15px")
                .html("No information was calculated for this token, since you used compute_buffer=False.");
    
            // Add dynamic behaviour: show the tooltip on hover, and also add lines to the two histograms
            tokenSpan.on('mouseover', function() {
                tooltipDiv.style('display', 'flex');
                tooltipDiv.style('position', 'fixed');
            })
            tokenSpan.on('mousemove', function(event) {
                tooltipDiv.style('left', `${event.clientX - tooltipWidth / 2}px`);
                tooltipDiv.style('top', `${event.clientY + 20}px`);
            });
            tokenSpan.on('mouseout', function() {
                tooltipDiv.style('display', 'none');
            });
            
        // If we're not hiding (because we only generated data for the bolded tokens), then create tooltip & add it on hover
        } else { 
    
            // First define the tooltip div (added to the parent element, i.e. it's a sibling of the token span)
            let tooltipHeight = isDfa ? 100 : (prevTokenActive ? 320 : 160);
            let tooltipWidth = isDfa ? 200 : (prevTokenActive ? 360 : 250);
            tooltipWidth = Math.max(tooltipWidth, 310 + 8 * (tok.length - 12));

            let tooltipDiv = parent.append("div")
                .attr("class", "tooltip")
                .style("height", tooltipHeight + "px")
                .style("width", tooltipWidth + "px")
                .style("align-items", "center")
                .style("text-align", "center");
    
            // Next, create a table container, to contain 2 tables: one with basics (acts & loss effect), one with per-token logits
            let tableContainer = tooltipDiv.append("div").attr("class", "table-container");
    
            // Creat the first table
            let firstTable = tableContainer.append("table");
            firstTable.append("tr").html(`<td class="right-aligned">Token</td><td class="left-aligned"><code>${tok.replace(/ /g, '&nbsp;')}</code> (${tokID})</td>`);
            firstTable.append("tr").html(`<td class="right-aligned">Dataset position</td><td class="left-aligned">${tokPosn}</td>`);
            firstTable.append("tr").html(`<td class="right-aligned">${isDfa ? 'DFA' : 'Feature activation'}</td><td class="left-aligned">${tokValue >= 0 ? '+' : ''}${tokValue.toFixed(3)}</td>`);
            if (!isDfa) {
                tableContainer.append("br");
            }
            
            // If previous token is active, we add logit table & loss info
            if (prevTokenActive) {
                // Loss effect of this feature, and probability change induced by feature
                firstTable.append("tr").html(`<td class="right-aligned">Logit effect (unnormalized)</td><td class="left-aligned">${logitEffect >= 0 ? '+' : ''}${logitEffect.toFixed(3)}</td>`);
                firstTable.append("tr").html(`<td class="right-aligned">Loss effect</td><td class="left-aligned">${lossEffect >= 0 ? '+' : ''}${lossEffect.toFixed(3)}</td>`);
                if (origProb !== null && newProb !== null) {
                    firstTable.append("tr").html(`<td class="right-aligned">Prob change from feature</td><td class="left-aligned">${(newProb*100).toFixed(2)}% → ${(origProb*100).toFixed(2)}%</td>`);
                }
                
                // Create container for top & bottom logits tables
                let logitsTableContainer = tableContainer.append("div").attr("class", "half-width-container")
    
                // Create the positive table, and fill it with values
                let posLogitsTable = logitsTableContainer.append("table").attr("class", "half-width")
                posLogitsTable.append("tr").html(`<td class="center-aligned" colspan="2">Pos logprob contributions</td>`);
                posToks.forEach((tok, index) => {
                    posLogitsTable.append("tr").html(`
                        <td class="right-aligned"><code>${tok.replace(/ /g, '&nbsp;')}</code></td>
                        <td class="left-aligned">${posVal[index] > 0 ? '+' : ''}${posVal[index].toFixed(3)}</td>
                    `);
                });
    
                // Create the negative table, and fill it with values
                let negLogitsTable = logitsTableContainer.append("table").attr("class", "half-width")
                negLogitsTable.append("tr").html(`<td class="center-aligned" colspan="2">Neg logprob contributions</td>`);
                negToks.forEach((tok, index) => {
                    negLogitsTable.append("tr").html(`
                        <td class="right-aligned"><code>${tok}</code></td>
                        <td class="left-aligned">${negVal[index] > 0 ? '+' : ''}${negVal[index].toFixed(3)}</td>
                    `);
                });
    
            // If previous token is not active, we add a message instead
            } else {
                if (!isDfa) {
                    tableContainer.append("div")
                        .style("font-size", "0.8em")
                        .html("Feature not active on prev token;<br>no predictions were affected.");
                }
            }
    
            // Add dynamic behaviour: show the tooltip on hover, and also add lines to the two histograms
            tokenSpan.on('mouseover', function() {
                tooltipDiv.style('display', 'flex');
                tooltipDiv.style('position', 'fixed');
                addLineHistogram(`actsHistogram-${featureIdx}`, 0, tok, tokValue);
                addLineHistogram(`logitsHistogram-${featureIdx}`, 0, tok, tokenLogit);
            })
            tokenSpan.on('mousemove', function(event) {
                tooltipDiv.style('left', `${event.clientX - tooltipWidth / 2}px`);
                tooltipDiv.style('top', `${event.clientY + 20}px`);
            });
            tokenSpan.on('mouseout', function() {
                tooltipDiv.style('display', 'none');
                removeLineHistogram(`actsHistogram-${featureIdx}`, 0);
                removeLineHistogram(`logitsHistogram-${featureIdx}`, 0);
            });
    
            // Add static behaviour: if required, then show the permanent line on the histograms, as shapes[1]
            if (permanentLine & isBold) {
                addLineHistogram(`actsHistogram-${featureIdx}`, 1, tok, tokValue);
                addLineHistogram(`logitsHistogram-${featureIdx}`, 1, tok, tokenLogit);
            }
        }
    }

    function addLineHistogram(histogramID, shapeIndex, tok, xValue) {
        // Updates histogram with a line (if the histogram exists)
        // shapeIndex 0 is for on-hover, 1 is for permanent line
        if (document.getElementById(histogramID)) {
            Plotly.relayout(histogramID, {
                [`shapes[${shapeIndex}].x0`]: xValue,
                [`shapes[${shapeIndex}].x1`]: xValue,
                [`shapes[${shapeIndex}].visible`]: true,
                [`annotations[${shapeIndex}].x`]: xValue,
                [`annotations[${shapeIndex}].text`]: `|${tok}|<br>${xValue.toFixed(3)}`,
                [`annotations[${shapeIndex}].visible`]: true,
            });
        }
    }
    
    function removeLineHistogram(histogramID, shapeIndex) {
        if (document.getElementById(histogramID)) {
            Plotly.relayout(histogramID, {
                [`shapes[${shapeIndex}].visible`]: false,
                [`annotations[${shapeIndex}].visible`]: false,
            });
        }
    }
}

function _setupSeqMultiGroupOthello(featureIdx, componentData, containerId) {
    const size = 225;
    const gap = 15;
    const cellSize = size / 8;
    const seqGroupsContainer = d3.select(`#${containerId}`)

    const movedColor = "mediumseagreen" // "orangered"
    const capturedColor = "yellowgreen" // "lightsalmon"
    
    componentData.forEach(seqGroup => {
        const {seqGroupData, seqGroupMetadata} = seqGroup;
        const {title, maxAct, maxLoss, nBoardsPerRow} = seqGroupMetadata;
        const width = (size+gap) * nBoardsPerRow - gap;

        seqGroupsContainer.append("h4").html(title);

        const boardContainer = seqGroupsContainer.append("div")
            .style("display", "flex")
            .style("flex-wrap", "wrap")
            .style("width", `${width}px`)
            .style("gap", `0px ${gap}px`);

        seqGroupData.forEach(({seqData, seqMetadata}, index) => {
            const boardWrapper = boardContainer.append("div")
                .style("width", `${size}px`)

            if ((index + 1) % nBoardsPerRow === 0) {
                boardContainer.append("div")
                    .style("flex-basis", "100%")
                    .style("height", "0");
            }

            drawBoard(seqData, maxAct, maxLoss, boardWrapper);
        });

        seqGroupsContainer.append("br");
    });
    
    function drawBoard(gameData, maxAct, maxLoss, container) {
        const { board, valid, move, captured, act, loss } = gameData;

        const header = container.append("div").attr("class", "highlight-header")
        header.append("span")
            .style("background-color", actColor(act, maxAct))
            .style("margin-right", "10px")
            .html(`ACT <b>${act.toFixed(3)}</b>`);
        header.append("span")
            .style("background-color", lossColor((+loss || 0), maxLoss))
            .style("margin-right", "10px")
            .html(`LOSS <b>${(+loss || 0).toFixed(3)}</b>`);
        // header.append("br")
        // const hoverText = header.append("span")
        //     .style("color", "#888")
        //     .style("background-color", "#eee")
        //     .html("Hover for prev board state")

        const svg = container.append("svg")
            .attr("width", size)
            .attr("height", size)
            .attr("style", "margin-bottom: 15px");

        const boardGroup = svg.append("g");

        function updateBoard(isLastBoard) {
            // Draw the cells
            const cells = boardGroup.selectAll("rect")
                .data(d3.cross(d3.range(8), d3.range(8)));

            cells.enter()
                .append("rect")
                .merge(cells)
                .attr("x", d => d[1] * cellSize)
                .attr("y", d => d[0] * cellSize)
                .attr("width", cellSize)
                .attr("height", cellSize)
                .attr("fill", d => {
                    if (d[0] === move[0] && d[1] === move[1]) return movedColor;
                    if (captured[d[0]][d[1]]) return capturedColor;
                    if (valid[d[0]][d[1]]) return "#777";
                    return "#ddd";
                })
                .attr("stroke", "white")
                .attr("stroke-width", 2.5);

            // Draw the pieces and square names
            const pieces = boardGroup.selectAll("g")
                .data(d3.cross(d3.range(8), d3.range(8)));

            const pieceGroups = pieces.enter()
                .append("g")
                .merge(pieces)
                .attr("transform", d => `translate(${(d[1] + 0.5) * cellSize}, ${(d[0] + 0.5) * cellSize})`);

            pieceGroups.each(function(d) {
                const g = d3.select(this);
                g.selectAll("*").remove();

                if (board[d[0]][d[1]] !== 0) {
                    // Add circle for piece
                    g.append("circle")
                        .attr("r", cellSize * 0.22)
                        .attr("fill", board[d[0]][d[1]] === 1 ? "black" : "white")
                        .attr("stroke", "black")
                        .attr("stroke-width", 1);
                } else {
                    // Add text for square name
                    g.append("text")
                        .attr("text-anchor", "middle")
                        .attr("dominant-baseline", "central")
                        .attr("font-size", cellSize * 0.4)
                        .attr("font-family", "monospace")
                        .attr("fill", valid[d[0]][d[1]] ? "white" : "black")
                        .text(`${String.fromCharCode(65 + d[0])}${d[1]}`);
                }
            });
        }  

        updateBoard(false)

        // updateBoard(false);
            
        // hoverText.on("mouseover", () => updateBoard(true)) // mousedown, mouseup
        //     .on("mouseout", () => updateBoard(false));          
    }
}

function setupSeqMultiGroup(featureIdx, componentData, containerId) {
    if (METADATA.othello) {
        _setupSeqMultiGroupOthello(featureIdx, componentData, containerId);
    } else {
        _setupSeqMultiGroupStandard(featureIdx, componentData, containerId);
    }
}

// ! Utils functions
// This is for functions which get re-used by multiple component-specific functions.

function lossColor(loss, maxLoss, opacity = 0.5) {
    // Interpolates between blue -> white -> red for [-maxLoss, 0, maxLoss]
    loss = Math.min(Math.max(loss, -maxLoss), maxLoss);

    return loss < 0
        ? d3.interpolate(`rgba(255,255,255,${opacity})`, `rgba(0,0,255,${opacity})`)(-loss / maxLoss)
        : d3.interpolate(`rgba(255,255,255,${opacity})`, `rgba(255,0,0,${opacity})`)(loss / maxLoss)
}

function actColor(value, maxValue, isDfa = false) {
    // Interpolates between white -> orange for [0, maxValue]
    value = Math.min(Math.max(value, 0), maxValue);
    return d3.interpolate("rgb(255,255,255)", isDfa ? "rgb(0,150,50)" : "rgb(255,140,0)")(value / maxValue);
}

function stringToId(str) {
    return str
        .toLowerCase()
        .replace(/['"]/g, '') // Remove single and double quotes
        .replace(/\s+/g, '-')
        .replace(/[^a-z0-9-]/g, '')
        .replace(/^-+|-+$/g, '')
        .replace(/-+/g, '-');
}

function createHistLine(color, width, height) {
    return {
        type: 'line', line: {color: color, width: width},
        x0: 0, x1: 0, xref: 'x',
        y0: 0, y1: height, yref: 'paper',
        visible: false,
    }
}

function createHistAnnotation() {
    return {
        text: '', showarrow: false,
        x: 0, xref: 'x', 
        y: 0.9, yref: 'paper',
        xshift: 3, align: 'left', xanchor: 'left',
        visible: false,
    }
}
  

// ! Last functions, for actually creating the vis

// Measures width of a key (for dynamically setting dropdown width)
function measureTextWidth(text, font) {
    const span = document.createElement("span");
    span.style.visibility = "hidden";
    span.style.position = "absolute";
    span.style.font = font;
    span.textContent = text;
    document.body.appendChild(span);
    const width = span.offsetWidth;
    document.body.removeChild(span);
    return width;
}

// * Called whenever we change the dropdown
// This function wraps around `setupPage` essentially, but also updates the dropdowns
function updateDropdowns(options) {

    // Get the current value of each dropdown
    const currentSelections = options.map((_, i) => d3.select(`#select-${i}`).property('value'));

    // Disable options that are not available based on the current selections
    options.forEach((opts, i) => {
        const select = d3.select(`#select-${i}`);
        select.selectAll('option').each(function() {
            const optionValue = d3.select(this).attr('value');
            const potentialKey = [...currentSelections.slice(0, i), optionValue, ...currentSelections.slice(i + 1)].join('|');
            // d3.select(this).attr('disabled', !DATA.hasOwnProperty(potentialKey) ? true : null);
        });
    });

    // Create vis using current dropdown values
    setupPage(currentSelections.join('|'));
}

const componentMap = {
    "featureTables": setupFeatureTables,
    "actsHistogram": setupActsHistogram,
    "logitsHistogram": setupLogitsHistogram,
    "logitsTable": setupLogitTables,
    "probeLogitsTables": setupProbeLogitsTables,
    "seqMultiGroup": setupSeqMultiGroup,
    "prompt": setupSeqMultiGroup,
};

// * Set up page, for a given key
function setupPage(key) {

    const promptVisMode = Object.keys(PROMPT_DATA).length > 0;

    // Empty the contents of the grid-container
    const gridContainer = d3.select(".grid-container").attr("style", `height: ${METADATA.height}px;`);
    gridContainer.selectAll('*').remove();

    if (!promptVisMode) {

        // * In feature-centric vis, keys = feature IDs, and we show that feature's data
        const feature = key;
        METADATA.layout.forEach((columnComponents, columnIdx) => {
            const columnWidth = METADATA.columnWidths[columnIdx];
            createColumn(columnComponents, columnIdx, columnWidth, feature)
        })

    } else {

        // * In prompt-centric vis, keys = stringified metrics, and we show a column for each feature
        const topFeaturesData = PROMPT_DATA[key];
        const columnComponents = METADATA.layout[0];
        const columnWidth = METADATA.columnWidths[0];
        topFeaturesData.forEach(({feature, title}, columnIdx) => {
            createColumn(columnComponents, columnIdx, columnWidth, feature, title);
        })
    }

    function createColumn(columnComponents, columnIdx, columnWidth, feature, title=null) {
        // Insert a column into the grid-container
        const column = gridContainer.append("div")
            .attr("id", `column-${columnIdx}`)
            .attr("class", "grid-column")
            .attr("style", `width: ${columnWidth ? columnWidth : "auto"}px`)

        // Add an optional title, as a div
        if (title) {
            column.append("div").html(title);
        }

        // Insert each of the features' components, in order
        columnComponents.forEach((componentName, componentIdx) => {
            var t0 = performance.now();

            const componentFn = componentMap[componentName];
            if (componentFn) {
                const containerId = `${componentName}-${feature}`
                column.append("div").attr("id", containerId).attr("class", componentName);
                componentFn(feature, DATA[feature][componentName], containerId);
            }
    
            var t1 = performance.now();
            console.log(`col ${columnIdx}, ${componentName}-${feature}: ${(t1 - t0).toFixed(1)} ms`);
        })
    }
    
}

// ! Create the vis when the page loads

document.addEventListener("DOMContentLoaded", function() {

    // Keys are either PROMPT_DATA.keys() if non-empty, else DATA.keys()
    const promptVisMode = Object.keys(PROMPT_DATA).length > 0;
    const allKeys = Object.keys(promptVisMode ? PROMPT_DATA : DATA);

    // The start key has already been user-defined, we need to check it's present in DASHBOARD_DATA
    if (!allKeys.includes(START_KEY)) {
        console.error(`No data available for key: ${START_KEY}`);
    }

    if (allKeys.length > 1) {
        const parsedKeys = allKeys.map(key => key.split("|"));

        // Create a structure to store options for each dropdown
        const options = Array.from({ length: parsedKeys[0].length }, () => new Set());

        // Populate options for each dropdown
        parsedKeys.forEach(parts => {
            parts.forEach((part, index) => options[index].add(part));
        });

        // Create the dropdowns
        options.forEach((opts, i) => {
            // We wrap each `select` element in a `.select` div, for styling reasons)
            const selectDiv = d3.select('#dropdown-container').append('div').attr('class', 'select');
            const selectElem = selectDiv.append('select').attr('id', `select-${i}`);
            let maxWidth = 0;

            opts.forEach(opt => {
                // Add this as an option
                selectElem.append('option')
                    .text(opt)
                    .attr('value', opt)
                    // .attr('selected', START_KEY.split('|')[i] === opt);

                // Calculate the width of this option, and possibly update the max width (for the select div)
                const width = measureTextWidth(opt, "1em system-ui");
                if (width > maxWidth) { maxWidth = width; }
            });

            selectElem.property('value', START_KEY.split('|')[i]);

            // Set the width of the select div to the max width + 45px (for the dropdown arrow)
            selectDiv.style('width', `${maxWidth + 45}px`);

            // Add event listener to update the visualization (and the selection options) when the dropdown changes
            selectElem.on('change', function() { updateDropdowns(options) });
        });
    }
    // ! Initial trigger
    setupPage(START_KEY);
})