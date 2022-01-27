<template>

    <button type="button" class="btn btn-link" style="padding:0rem 0.75rem 0rem 0rem; font-size:1.2rem" data-toggle="modal" :data-target="'#walkthroughModal'+idModal">
    <i class="bi bi-journal-text"></i>
    </button>

    <div class="modal fade" :id="'walkthroughModal'+idModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
        <div class="modal-header">
            <h5 class="modal-title" id="walkthroughModalLabel"> Walkthrough </h5>
            <button type="button" class="close" data-dismiss="modal">&times;</button>
        </div>
        <div class="modal-body">
            <h6>Steps to build and simulate an electricity distribution network</h6>
            <ol>
                <li style="padding: 0.2rem 0rem">
                    Build the network. Leave the default options in the “Electricity distribution network parameters” section or modify them to change the network.
                    The integrated MV-LV 11kV urban network represents a town of around 20k households. 
                    You can choose up to 5 LV networks - this is equivalent to simulating 5 neighbourhoods in detail (2 examples of LV networks/neighbourhoods, from the LVNS project, are shown on top of the figure below.
                </li>
                <li style="padding: 0.2rem 0rem">
                    Assign data to the network.
                    Use available demand and generation profiles or upload your own data in “Demand and generation profiles” section.
                    You can choose to have industrial solar installations and fast charging stations.
                    Then you can choose the mix and penetration levels of green technologies assigned to the households, which are shown as dots on the LV networks.
                </li>
                <li style="padding: 0.2rem 0rem">
                    Run your simulation.
                    Click submit and wait for up to a minute for the results.
                    You can save the figures and download some of output data as csv.
                    The results include information on the electricity network status, specifically voltage and thermal constraints for the MV and LV networks. 
                </li>
            </ol>
            If you want to learn more about the details of this tool, including network characteristics and data sources, you can find more information in the project repositories.
            <img src="../assets/images/11kv_urban.png" style="max-width: 100%; min-width: 75%; padding: 1rem 0rem">
        </div>
        <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
        </div>
        </div>
    </div>
    </div>

</template>

<script>
    import { v4 as uuidv4 } from "uuid";
    export default {
        data() {
            return { 
                showModal: false,
                idModal: uuidv4(),
            }
        },
    }
</script>