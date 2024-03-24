// ! Using `DATA.actsHistogramData` to fill in the activations histogram

function createActsLine(color, width, height) {
    return {
        type: 'line', line: {color: color, width: width},
        x0: 0, x1: 0, xref: 'x',
        y0: 0, y1: height, yref: 'paper',
        visible: false,
    }
}

function createActsAnnotation() {
    return {
        text: '', showarrow: false,
        x: 0, xref: 'x', 
        y: 0.9, yref: 'paper',
        xshift: 3, align: 'left', xanchor: 'left',
        visible: false,
    }
}

function setupActsHistogram(histId, histData) {

    // Create layout. This has 2 lines with annotations, both initially set to invisible (first is for dynamic on-hover
    // and second is for static, for the prompt-centric vis). We need to define them both so that when JavaScript alters
    // 'shapes[0]' or 'shapes[1]' we get no error
    var layout = {
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: {
            gridcolor: '#eee',
            zerolinecolor: '#eee',
            tickvals: histData.ticks,
            range: [0, 1.2 * Math.max(...histData.x)],
        },
        yaxis: {gridcolor: '#eee', zerolinecolor: '#eee'},
        barmode: 'relative',
        bargap: 0.01,
        showlegend: false,
        margin: {l: 50, r: 25, b: 25, t: 25, pad: 4},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        responsive: true,
        shapes: [createActsLine('black', 2, 1.0), createActsLine('black', 1, 0.9)],
        annotations: [createActsAnnotation(), createActsAnnotation()],
    };

    // Create traces (this is simple)
    var traces = [{
        x: histData.x,
        y: histData.y,
        type: 'bar',
        marker: {color: histData.colors},
    }];

    // Plot the histogram
    Plotly.newPlot(histId, traces, layout, {responsive: true, displayModeBar: false});

    // Maybe add title (if the title already existed then we update it, otherwise we create a new one)
    if(histData.title) {
        var histogramElement = document.getElementById(histId);
        var existingTitleElement = histogramElement.previousSibling;
        if (existingTitleElement && existingTitleElement.tagName === 'H4') {
            existingTitleElement.innerHTML = histData.title;
        } else {
            var titleElement = document.createElement('h4');
            titleElement.innerHTML = histData.title;
            histogramElement.parentNode.insertBefore(titleElement, histogramElement);
        }
    }
}

// 'actsHistogramData' is a dictionary mapping suffixes to histogram data (to make each histogram unique)
// We iterate over it, and create a histogram for each one
Object.entries(DATA.actsHistogramData).forEach(([suffix, histData]) => {
    var t0 = performance.now();

    histId = `histogram-acts-${suffix}`;
    setupActsHistogram(histId, histData);

    var t1 = performance.now();
    console.log(`HTML for ${histId} generated in ${(t1 - t0).toFixed(1)} ms`);
});



