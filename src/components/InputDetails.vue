<template>

    <button type="button" class="btn btn-link" data-toggle="modal" data-target="#inputDetailsModal">
    <i class="bi bi-info-circle"></i>
    </button>

    <div class="modal fade" id="inputDetailsModal" tabindex="-1">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <h5 class="modal-title" id="inputDetailsModalLabel"> {{ inputName }} </h5>
            <button type="button" class="close" data-dismiss="modal">&times;</button>
        </div>
        <div class="modal-body">
            <p>{{ inputInfo }}</p>
            <p>{{ inputValues }}</p>
        </div>
        <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
        </div>
        </div>
    </div>
    </div>

</template>

<script>
    export default {
        props: ["inputName", "inputInfo", "inputValues"],
        data() {
            return { showModal: false}
        }
    }
</script>