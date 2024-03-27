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

function generateTokHtmlElement(
    parent, tok, tokID, uColor, bgColor, isBold, featAct, tokenLogit, lossEffect, posToks, posVal, negToks, negVal, permanentLine, hide, idSuffix,
) {
    // Figure out if previous token was active (this affects the construction of the tooltip)
    let prevTokenActive = posToks.length + negToks.length > 0;

    // Create the token span (this will contain just the token, not the hoverdata)
    let tokenSpan = parent.append("span")
        .attr("class", "hover-text");

    // Get the histogram ids
    let histogramActsID = `histogram-acts-${idSuffix}`;
    let histogramLogitsID = `histogram-logits-${idSuffix}`;

    // Put actual token in the tokenSpan object (i.e. the thing we see without hovering)    
    tokenSpan.append("span")
        .attr("class", "token")
        .style("background-color", bgColor)
        .style("border-bottom", `4px solid ${uColor}`)
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
            .style("font-size", "0.85em")
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
        let tooltipHeight = prevTokenActive ? 270 : 160;
        let tooltipWidth = prevTokenActive ? 350 : 250;
        let tooltipDiv = parent.append("div")
            .attr("class", "tooltip")
            .style("height", tooltipHeight + "px")
            .style("width", tooltipWidth + "px")
            .style("align-items", "center")
            .style("text-align", "center");

        // Next, create a table container, to contain 2 tables: one with basics (acts & loss effect), one with per-token logits
        let tableContainer = tooltipDiv.append("div").attr("class", "table-container");

        // Creat the first table
        let tokRow = `<td class="right-aligned">Token</td><td class="left-aligned"><code>${tok}</code> (${tokID})</td>`;
        let featActRow = `<td class="right-aligned">Feature activation</td><td class="left-aligned">${featAct >= 0 ? '+' : ''}${featAct.toFixed(3)}</td>`;
        let lossRow = `<td class="right-aligned">Loss contribution</td><td class="left-aligned">${lossEffect >= 0 ? '+' : ''}${lossEffect.toFixed(3)}</td>`;
        let firstTable = tableContainer.append("table");
        firstTable.append("tr").html(tokRow);
        firstTable.append("tr").html(featActRow);
        firstTable.append("tr").html(lossRow);
        tableContainer.append("br");
        
        // If previous token is active, we add logit table
        if (prevTokenActive) {
            
            // Create container for top & bottom logits tables
            let logitsTableContainer = tableContainer.append("div").attr("class", "half-width-container")

            // Create the positive table, and fill it with values
            let posLogitsTable = logitsTableContainer.append("table").attr("class", "half-width")
            posLogitsTable.append("tr").html(`<td class="center-aligned" colspan="2">Pos logit contributions</td>`);
            posToks.forEach((tok, index) => {
                posLogitsTable.append("tr").html(`
                    <td class="right-aligned"><code>${tok}</code></td>
                    <td class="left-aligned">${posVal[index] > 0 ? '+' : ''}${posVal[index].toFixed(3)}</td>
                `);
            });

            // Create the negative table, and fill it with values
            let negLogitsTable = logitsTableContainer.append("table").attr("class", "half-width")
            negLogitsTable.append("tr").html(`<td class="center-aligned" colspan="2">Neg logit contributions</td>`);
            negToks.forEach((tok, index) => {
                negLogitsTable.append("tr").html(`
                    <td class="right-aligned"><code>${tok}</code></td>
                    <td class="left-aligned">${negVal[index] > 0 ? '+' : ''}${negVal[index].toFixed(3)}</td>
                `);
            });

        // If previous token is not active, we add a message instead
        } else {
            tableContainer.append("div")
                .style("font-size", "0.85em")
                .html("Feature not active on prev token;<br>no predictions were affected.");
        }

        // Add dynamic behaviour: show the tooltip on hover, and also add lines to the two histograms
        tokenSpan.on('mouseover', function() {
            tooltipDiv.style('display', 'flex');
            tooltipDiv.style('position', 'fixed');
            addLineHistogram(histogramActsID, 0, tok, featAct);
            addLineHistogram(histogramLogitsID, 0, tok, tokenLogit);
        })
        tokenSpan.on('mousemove', function(event) {
            tooltipDiv.style('left', `${event.clientX - tooltipWidth / 2}px`);
            tooltipDiv.style('top', `${event.clientY + 20}px`);
        });
        tokenSpan.on('mouseout', function() {
            tooltipDiv.style('display', 'none');
            removeLineHistogram(histogramActsID, 0);
            removeLineHistogram(histogramLogitsID, 0);
        });

        // Add static behaviour: if required, then show the permanent line on the histograms, as shapes[1]
        if (permanentLine & isBold) {
            addLineHistogram(histogramActsID, 1, tok, featAct);
            addLineHistogram(histogramLogitsID, 1, tok, tokenLogit);
        }
    }
}

Object.entries(DATA.tokenData).forEach(([seqGroupID, seqGroupData]) => {
    const t0 = performance.now();

    // Find the sequence group container (this is the only thing that already exists)
    const seqGroupContainer = d3.select(`#${seqGroupID}`);

    // Empty this container of title & all sequences
    seqGroupContainer.selectAll('*').remove();

    // If title exists, add the title to this sequence group (we need .html not .text, because it could have <br>)
    if ("title" in seqGroupData) {
        seqGroupContainer.append('h4').html(seqGroupData.title);
    }

    // Get the ID suffix for this sequence group
    const idSuffix = seqGroupData.idSuffix;

    // Select all sequences (initially empty), and then add our sequences & bind them to the elems in seqGroupData.data
    const seqGroup = seqGroupContainer.selectAll('.seq')
        .data(seqGroupData.data)
        .enter()
        .append('div')
        .attr('class', 'seq');

    // For each sequence, we iterate over & add all its tokens
    seqGroup.each(function(seqData) {

        // Get `seq`, which is the individual `seq` element in seqGroup (i.e. the sequence container)
        const seq = d3.select(this);

        // Iterate through each token data dict in `seqData`, and add this token to the sequence
        seqData.forEach(tokData => {

            generateTokHtmlElement(
                seq,                                // object we'll append the token to
                tokData.tok,                        // string token
                tokData.tokID,                      // token ID (shown on hover, after PR request)
                tokData["uColor"] || "#fff",        // underline color (derived from loss effect)
                tokData["bgColor"] || "#fff",       // background color (derived from feature activation)
                tokData["isBold"] || false,         // is this token bolded?
                tokData["featAct"] || 0.0,          // feature activation at this token (used for acts histogram line)
                tokData["tokenLogit"] || 0.0,       // raw logit effect on this token (used for logits histogram line)
                tokData["lossEffect"] || 0.0,       // effect on loss (used for histogram line), if prev token active
                tokData["posToks"] || [],           // most-positive tokens (strings)
                tokData["posVal"] || [],            // most-positive token values
                tokData["negToks"] || [],           // most-negative tokens (strings)
                tokData["negVal"] || [],            // most-negative token values
                tokData["permanentLine"] || false,  // do we show a permanent line on histogram?
                tokData["hide"] || false,           // do we suppress hoverdata for this token?
                idSuffix,                           // suffix for the histogram ID which the hoverline will be added to
            );
        });
    });

    const t1 = performance.now(); // End timing
    console.log(`HTML for ${seqGroupID} generated in ${(t1 - t0).toFixed(1)} ms`);
});