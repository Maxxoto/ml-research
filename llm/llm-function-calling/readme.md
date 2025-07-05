# LLM Function Calling
=======================

## Overview
-----------

A simple server that uses a Large Language Model (LLM) to respond to user inquiries. The LLM is capable of calling external functions to gather information and provide more accurate responses.

## Features
------------

*   Uses a Large Language Model (LLM) to respond to user inquiries
*   Capable of calling external functions to gather information
*   Supports streaming responses from the LLM
*   Includes a simple web interface for testing

## Available Tools
-------------------

### get_exchange_rate

*   **Description:** Function to get exchange rate currency
*   **Usage:** `get_exchange_rate: <from_currency> to <to_currency>`
*   **Example:** `get_exchange_rate: USD to IDR`

### Start the Server

```bash
node index.js
```

## Usage
-----

### Step 1: Open the Web Interface

Open a web browser and navigate to `http://localhost:3000`

### Step 2: Enter a Question or Prompt

Enter a question or prompt in the input field and press enter

### Step 3: View the Response

The LLM will respond with an answer, which may include calls to external functions

## API
----

### Endpoints

*   **GET /chat**: Responds with a JSON object containing the LLM's response to the user's inquiry
*   **GET /health**: Responds with a simple "OK" message to indicate that the server is healthy
*   **GET /index.html**: Responds with the HTML for the web interface

## License
-------

This project is licensed under the MIT License. See the LICENSE file for details.
