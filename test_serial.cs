using System;
using System.IO.Ports; 
using System.Diagnostics; 


public class SerialTest 
{
    public static void Main() 
    {
        /*
        FPGA Read Write Process
            - load bytes into local bytes array 
            - stream to FPGA using port.write()
            - read in serial input buffer using ReadAvailable()
        */

        // Variables for loading in the input byte stream
        string inputFileName = @"uart_input_bytes.txt";
        string[] hexBytes = System.IO.File.ReadAllText(inputFileName).Split('\n');
        Byte[] fpgaStream = new Byte [hexBytes.Length];
        int i = 0; 

        //Variables for the serial port reading/writing
        string fpgaBytesRead; 
        string portName = "COM4";
        int    baudRate = 6000000;
        int    dataBits = 8;
        int    readInterval = 1; 
        var    stopBit  = System.IO.Ports.StopBits.One;  
        var    parity   = System.IO.Ports.Parity.None;

        SerialPort fpgaComPort = new SerialPort(portName,baudRate,parity,dataBits,stopBit); 
        // fpgaComPort.Handshake = System.IO.Ports.Handshake.RequestToSend;

        // Helper Variables for serial r/w byte counting
        int numOutputsExpected = 518*518*2; 
        int numOutputsRecieved = 0; 
        int bytesRemaining = 0; 

        string outputFileName = @"fpgaOut.txt";
        System.IO.StreamWriter outputFile = new System.IO.StreamWriter(outputFileName);
        System.IO.StreamWriter testFile = new System.IO.StreamWriter("test.txt");
        
        /* Load all FPGA Input Bytes into a Byte[] struct */ 
        int numFPGAStreamElements = 0; 

        for (i = 0; i < hexBytes.Length; i++)
        {          
             try 
             {
                 Byte hexValue = Convert.ToByte(hexBytes[i]);
                 fpgaStream[i] = hexValue; 
             }
            catch (Exception) { Console.WriteLine("Conversion Failed!");}
            numFPGAStreamElements++;
        }

        Console.WriteLine("Loaded Input Bytes!");

        /* Send/Recieve Bytes from FPGA COM Ports */
        fpgaComPort.Open();
        // fpgaComPort.Write("\n");

        // Clean up the input/output buffers
        fpgaComPort.DiscardInBuffer();
        fpgaComPort.DiscardOutBuffer();
        fpgaComPort.BaseStream.Flush();

        Stopwatch timer = new Stopwatch(); 
        timer.Start();

        foreach(Byte outputByte in fpgaStream)
        {
            Byte[] writeByte = new byte [1];
            Byte [] outputFileByte = new byte [1];
            string hexStr;

            writeByte[0] = outputByte; 

            fpgaComPort.Write(writeByte,0,1);

            /* Allow 1ms to let Serial Input buffer fill with data */
            //System.Threading.Thread.Sleep(readInterval);

            Byte[] bufData = new byte [fpgaComPort.BytesToRead];
            fpgaComPort.Read(bufData,0, bufData.Length);

            Console.WriteLine("Number of Bytes Recieved: {0}", bufData.Length);

            for (int j = 0; j < bufData.Length; j++)
            {
                outputFileByte[0] = bufData[j];
                hexStr = BitConverter.ToString(outputFileByte).Replace("-", string.Empty);
                outputFile.WriteLine(hexStr);
            }

            numOutputsRecieved += bufData.Length; 
            Console.WriteLine("Total FPGA Outputs Recieved = {0}", numOutputsRecieved);

        }
        
        /* Read the leftover FPGA Data */ 
        bytesRemaining = numOutputsExpected-numOutputsRecieved; 

        for (int p = 0; p < bytesRemaining; p++)
        {
            //System.Threading.Thread.Sleep(readInterval);

            Byte[] bufData = new byte [fpgaComPort.BytesToRead];
            Byte[] thing = new byte[1];

            string hexStr; 
            fpgaComPort.Read(bufData,0, bufData.Length);

            Console.WriteLine("Number of Bytes Recieved: {0}", bufData.Length);

            for (int s = 0; s < bufData.Length; s++)
            {
                thing[0] = bufData[s]; 

                hexStr = BitConverter.ToString(thing).Replace("-", string.Empty);
                outputFile.WriteLine(hexStr);
            }

            numOutputsRecieved += bufData.Length; 
            Console.WriteLine("Total FPGA Outputs Recieved = {0}", numOutputsRecieved);
        }
        fpgaComPort.Close();

        timer.Stop(); 
        Console.WriteLine("Execution Time: {0} ms", timer.ElapsedMilliseconds);

        return; 
    }
}