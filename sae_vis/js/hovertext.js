document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('.hover-text').forEach(elem => {
        elem.addEventListener('mouseover', function() {
            const tooltipId = this.dataset.tooltipId;
            const tooltip = document.getElementById(tooltipId);
            if (tooltip) {
                const rect = this.getBoundingClientRect();
                const leftPosition = rect.left;
                const topPosition = rect.top;

                const tooltipWidth = parseInt(tooltip.style.width, 10);
                const tooltipHeight = parseInt(tooltip.style.height, 10);

                // const hoverRect = elem.getBoundingClientRect();
                // const hoverWidth = hoverRect.width;

                tooltip.style.left = `${leftPosition-tooltipWidth/2}px`;
                tooltip.style.top = `${topPosition-tooltipHeight-20}px`;
                tooltip.style.display = 'flex';
                tooltip.style.position = 'fixed'; // Ensure absolute positioning relative to viewport
            }
        });

        elem.addEventListener('mouseout', function() {
            const tooltipId = this.dataset.tooltipId;
            const tooltip = document.getElementById(tooltipId);
            if (tooltip) {
                tooltip.style.display = 'none';
            }
        });
    });
});